import os
from cProfile import label
from pprint import pprint

from utils import metrics
from utils.instatiate_utils import get_model

os.environ["WANDB_MODE"] = "disabled"
import time
from dataclasses import dataclass

import hydra
import torch
from ignite.utils import manual_seed
from omegaconf import OmegaConf
from spikingjelly.activation_based.functional import reset_net

import wandb
from attacker import GumbelAttacker, GumbelFrameAttacker
from attacks.ours.probability_space.frame_generator import FrameGenerator
from attacks.ours.probability_space.probability_attack import (
    ProbabilityAttacker,
    ProbabilityFrameAttacker,
)
from utils.general_utils import *
from utils.instatiate_utils import get_dataloaders, get_event_generator
from utils.optimizer import CosineAnnealing, get_lr_scheduler, get_optimizer


@hydra.main(config_path="./config", config_name="default", version_base=None)
def main(dictconfig: DictConfig):
    cfg: dict = OmegaConf.to_container(dictconfig, resolve=True)  # type:ignore
    # pprint(cfg, depth=3)
    manual_seed(cfg["seed"])

    wandb.init(
        config=cfg,
        project=cfg["project"],
        entity=cfg["entity"],
        name=cfg["name"],
    )
    wandb.define_metric("*", step_metric="global_step")

    # prepare device, dataset and model
    device = get_device(cfg["gpu_idx"])
    cfg["dataset"]["batch_size"] = cfg["num_pic"]
    train_loader, test_loader, _ = get_dataloaders(
        **cfg["dataset"],
        transform=cfg["transform"],
    )
    attack_frame, true_lables = next(iter(test_loader))
    attack_frame = attack_frame.to(device)
    true_lables = true_lables.to(device)
    model = get_model(cfg, device)
    attack_frame, true_lables = remove_misclassified_samples(
        model, attack_frame, true_lables, device
    )
    attack_frame = attack_frame.transpose(0, 1)
    num_attack_samples = len(attack_frame)
    print(f"Number of attack samples: {num_attack_samples}")
    total_metrics = TotalMetrics(num_attack_samples=num_attack_samples)

    for i, (frame, label) in enumerate(zip(attack_frame, true_lables)):
        probability_attacker = ProbabilityFrameAttacker(
            attack_cfg=cfg["attack"], binary_frame=frame, device=device
        )
        optimizer_alpha = get_optimizer(
            probability_attacker.parameters(), **cfg["optimizer"]
        )
        lr_scheduler = get_lr_scheduler(optimizer_alpha, **cfg["scheduler"])
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"])  # type:ignore
        main_attacker = GumbelFrameAttacker(
            cfg=cfg,
            probability_attacker=probability_attacker,
            target_label=label,
            orginal_frame=frame,
            optimizer_alpha=optimizer_alpha,
            model=model,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
        attack_metrics = AttackMetrics()
        epoch = 0
        for epoch in range(1, cfg["max_iteration"] + 1):
            one_step_metrics, results = main_attacker.attack_one_step(epoch=epoch)
            preds = torch.argmax(results["logits"], dim=-1)
            is_success_attack = is_success(preds, label, cfg["targeted"])
            attack_metrics.update_metrics(
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                **one_step_metrics,
            )

            if epoch % cfg["log_loss_interval"] == 0:
                attack_metrics.upload(
                    step=epoch,
                    name="",
                    image_id=i,
                )
                attack_metrics.reset()

            if is_success_attack:
                print(f"[succ] Image:{i}, target:{label.item()}, iteration:{epoch}")
                total_metrics.update_metrics(frame, results["attack_frame"], True)
                break

        if epoch == cfg["max_iteration"]:
            print(f"[fail] Image:{i}, target:{label.item()}, iteration:{epoch}")
            total_metrics.update_metrics(frame, results["attack_frame"], False)

    total_metrics.finalize_metrics()


class TotalMetrics:
    def __init__(self, num_attack_samples) -> None:
        self.total_success_num = 0
        self.num_attack_samples = num_attack_samples
        self.success_l0_list = []
        self.failed_l0_list = []
        self.success_l1_list = []
        self.failed_l1_list = []
        self.success_l2_list = []
        self.failed_l2_list = []

    @torch.no_grad()
    def update_metrics(self, original_frame, attack_frame, is_success_attack: bool):
        self.total_success_num += 1 if is_success_attack else 0
        original_frame = original_frame.unsqueeze(1)
        l0 = torch.norm((attack_frame - original_frame), p=0)
        l1 = torch.norm((attack_frame - original_frame), p=1)
        # l2 = torch.norm((attack_frame - original_frame), p=2)
        l2 = torch.mean((attack_frame - original_frame) ** 2)
        if is_success_attack:
            self.success_l0_list.append(l0)
            self.success_l1_list.append(l1)
            self.success_l2_list.append(l2)
        else:
            self.failed_l0_list.append(l0)
            self.failed_l1_list.append(l1)
            self.failed_l2_list.append(l2)

    @torch.no_grad()
    def finalize_metrics(
        self,
    ):
        success_rate = self.total_success_num / self.num_attack_samples
        if len(self.success_l0_list) == 0:
            success_l0 = 0
            success_l1 = 0
            success_l2 = 0

        else:
            success_l0 = torch.mean(torch.stack(self.success_l0_list))
            success_l1 = torch.mean(torch.stack(self.success_l1_list))
            success_l2 = torch.mean(torch.stack(self.success_l2_list))

        if len(self.failed_l0_list) == 0:
            failed_l0 = 0
            failed_l1 = 0
            failed_l2 = 0
        else:
            failed_l0 = torch.mean(torch.stack(self.failed_l0_list))
            failed_l1 = torch.mean(torch.stack(self.failed_l1_list))
            failed_l2 = torch.mean(torch.stack(self.failed_l2_list))
        result = {
            "num_attack_samples": self.num_attack_samples,
            "success_rate": success_rate,
            "success_l0": success_l0,
            "success_l1": success_l1,
            "success_l2": success_l2,
            "failed_l0": failed_l0,
            "failed_l1": failed_l1,
            "failed_l2": failed_l2,
        }
        wandb.log(result)
        pprint(result)


@dataclass
class AttackMetrics:
    """
    A class to track and update attack metrics.

    Attributes:
    - total_loss (float): The total loss.
    - regular_loss (float): The regular loss.
    - main_loss (float): The main loss.

    Methods:
    - update_metrics(total_loss, regular_loss, main_loss, gradient_accumulation_steps): Updates the metrics by adding the provided values.
    - upload(step, name, image_id, attacker): Uploads the metrics to the logging platform.
    - reset(): Resets the metrics to zero.
    """

    total_loss: float = 0
    regular_loss: float = 0
    main_loss: float = 0

    def update_metrics(
        self, total_loss, regular_loss, main_loss, gradient_accumulation_steps
    ):
        """
        Updates the metrics by adding the provided values.

        Parameters:
        - total_loss (float): The total loss.
        - regular_loss (float): The regular loss.
        - main_loss (float): The main loss.
        - gradient_accumulation_steps (int): The number of gradient accumulation steps.

        Returns:
        None
        """
        self.total_loss += total_loss / gradient_accumulation_steps
        self.regular_loss += regular_loss / gradient_accumulation_steps
        self.main_loss += main_loss / gradient_accumulation_steps

    def upload(self, step: int, name: str, image_id: int):
        """
        Uploads the metrics to the logging platform.

        Parameters:
        - step (int): The step number.
        - name (str): The name of the metric.
        - image_id (int): The ID of the image.

        Returns:
        None
        """
        metrics = {
            f"total_loss/total_loss_{image_id}": self.total_loss,
            f"regular_loss/regular_loss_{image_id}": self.regular_loss,
            f"main_loss/main_loss_{image_id}": self.main_loss,
            "global_step": step,
        }
        wandb.log(metrics)

    def reset(self):
        """
        Resets the metrics to zero.

        Returns:
        None
        """
        self.total_loss = 0
        self.regular_loss = 0
        self.main_loss = 0


def remove_misclassified_samples(
    model: torch.nn.Module,
    frame: torch.Tensor,
    lables: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes misclassified samples from the given frame and labels based on the predictions made by the model.

    Args:
        model (torch.nn.Module): The model used for classification.
        frame (torch.Tensor): The input frame containing samples.
        lables (torch.Tensor): The corresponding labels for the samples in the frame.
        device (torch.device): The device on which the model and tensors are located.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the filtered frame and labels.
    """
    reset_net(model)
    logits = model(frame.to(device)).mean(dim=0)
    preds = torch.argmax(logits, dim=-1)
    return frame[:, preds == lables, ...], lables[preds == lables]


if __name__ == "__main__":
    main()
