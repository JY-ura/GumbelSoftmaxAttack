import os
from pprint import pprint

from utils.instatiate_utils import get_model

os.environ["WANDB_MODE"] = "online"
import time

import hydra
import torch
from ignite.utils import manual_seed
from omegaconf import OmegaConf, DictConfig

import wandb
from attacker import GumbelAttacker
from attacks.ours.probability_space.frame_generator import FrameGenerator
from attacks.ours.probability_space.probability_attack import ProbabilityAttacker
from utils.general_utils import *
from utils.instatiate_utils import get_dataloaders, get_event_generator
from utils.metrics import Metrics, TotalMetrics
from utils.optimizer import CosineAnnealing, get_lr_scheduler, get_optimizer


@hydra.main(config_path="./config", config_name="default", version_base=None)
def main(dictconfig: DictConfig):
    cfg: dict = OmegaConf.to_container(dictconfig, resolve=True)  # type:ignore
    manual_seed(cfg["seed"])
    cfg["use_grad_scaling"] = cfg["use_grad_scaling"] if cfg["use_amp"] else False
    assert cfg["dataset"]["data_type"] == "event"

    # init one run for all image
    wandb.init(
        config=cfg,
        project=cfg["project"],
        entity=cfg["entity"],
        name=cfg["name"],
        reinit=True,
    )
    wandb.define_metric("*", step_metric="global_step")

    # prepare device, dataset and model
    device = get_device(cfg["gpu_idx"])
    _, _, test_data = get_dataloaders(
        **cfg["dataset"],
        transform=cfg["transform"],
    )
    model = get_model(cfg, device)

    # init some generator
    event_generator = get_event_generator(cfg)
    frame_processor = FrameGenerator(
        split_by=cfg["dataset"]["split_by"],
        frame_number=cfg["dataset"]["frame_number"],
        frame_size=cfg["dataset"]["frame_size"],
    )

    (
        correct_num,
        correct_events,
        true_labels,
        true_values,
        true_indices,
        true_frame_list,
    ) = remove_misclassification(cfg, test_data, model, frame_processor, device)
    target_labels = get_target_label(cfg, true_labels, device)
    auxiliary_event_dict: dict = generate_auxiliary_samples_for_each_sample(
        cfg, true_labels, correct_events
    )

    # Prepare a sample for each sample to be attacked.
    metrics = Metrics(total_num_correct=correct_num)
    start = time.time()
    for i, (event, true_label, target_label, true_frame, true_indice) in enumerate(
        zip(correct_events, true_labels, target_labels, true_frame_list, true_indices)
    ):
        target_label_for_addition_position = get_target_label_for_add_position(
            add_position_label_mode=cfg["attack"]["add_position_label_mode"],
            target_label=target_label,
            num_class=cfg["dataset"]["num_class"],
        )
        alpha_dict = {
            "events": event,
            "device": device,
            "init_alpha_mode": cfg["attack"]["init_alpha_mode"],
            "event_dict": auxiliary_event_dict,
            "target_label": target_label_for_addition_position,
            "target_position_ratio": cfg["attack"]["target_position_ratio"],
            "true_label": true_label,
            "num_class": cfg["dataset"]["num_class"],
        }
        (
            true_value_for_attack,
            hard_true_value,
            probability_attacker,
            optimizer_alpha,
            lr_scheduler,
            scaler,
            temperture_tau_scheduler,
        ) = prepare_attack(
            cfg=cfg,
            alpha_dict=alpha_dict,
            event_generator=event_generator,
            frame_processor=frame_processor,
        )

        # wandb.watch(probability_attacker, log="all", log_freq=10)

        # initial an attacker
        attacker = GumbelAttacker(
            cfg=cfg,
            probability_attacker=probability_attacker,
            target_label=target_label,
            orginal_value=true_value_for_attack,
            optimizer_alpha=optimizer_alpha,
            model=model,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
        )

        # start attack
        attack_metrics = TotalMetrics()
        one_step_metrics = None
        epoch = 0
        for epoch in range(1, cfg["max_iteration"] + 1):
            # attack one step
            one_step_metrics, results = attacker.attack_one_step(epoch=epoch)

            results["hard_event_indices"] = true_indice.repeat_interleave(
                cfg["attack"]["sample_num"], dim=0
            )

            # judge if success
            preds = torch.argmax(results["logits"], dim=-1)
            if is_success(preds, target_label, cfg["targeted"]):
                metrics.update_metrics(
                    cfg=cfg,
                    preds=preds.cpu(),
                    target_label=target_label.cpu(),
                    results=results,
                    true_value=hard_true_value,
                    iters=epoch,
                    i=i,
                    one_step_metrics=one_step_metrics,
                    true_frame=true_frame,
                    true_label=true_label,
                )
                metrics.event_number_orginal_list.append(event["p"].shape[0])

                print(
                    f"[succ] Image:{i}, target:{target_label.item()}, iteration:{epoch}"
                )
                break

            # update attack_metrics
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

            temperture_tau_scheduler.step(iters=epoch)
            model.zero_grad()
            torch.cuda.empty_cache()
        else:
            print(f"[fail] Image:{i}, target:{target_label.item()}, iter:{epoch}")

        torch.cuda.synchronize()
        del probability_attacker, optimizer_alpha

    end = time.time()
    metrics.finalize_metrics(start=start, end=end, cfg=cfg)


def prepare_attack(
    cfg: dict,
    alpha_dict: dict,
    event_generator,
    frame_processor,
):
    """
    Prepares the attack by initializing the necessary components and parameters.

    Args:
        cfg (dict): Configuration dictionary containing attack settings.
        alpha_dict (dict): Dictionary containing alpha values.
        event_generator: Event generator object.
        frame_processor: Frame processor object.

    Returns:
        Tuple: A tuple containing the following elements:
            - true_value_for_attack: True value for the attack.
            - hard_true_value: Hard true value for the attack.
            - event2value_attacker: event2value_attacker object.
            - optimizer_alpha: Optimizer for alpha values.
            - lr_scheduler: Learning rate scheduler.
            - scaler: Gradient scaler.
            - temperture_tau_scheduler: Temperature tau scheduler.
    """
    event2value_attacker = ProbabilityAttacker(
        attack_cfg=cfg["attack"],
        alpha_dict=alpha_dict,
        event_generator=event_generator,
        frame_processor=frame_processor,
    )

    hard_true_value, true_value_for_attack = get_true_values(
        event2value_attacker.alpha.clone(), use_soft_event=cfg["use_soft_event"]
    )

    # preprocess
    optimizer_alpha = get_optimizer(
        event2value_attacker.parameters(), **cfg["optimizer"]
    )
    lr_scheduler = get_lr_scheduler(optimizer_alpha, **cfg["scheduler"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"])  # type:ignore
    temperture_tau_scheduler = CosineAnnealing(
        optimizer=event_generator,
        initial_value=cfg["attack"]["min_tau"],
        final_value=cfg["attack"]["max_tau"],
        decay_step=cfg["attack"]["decay_step"],
    )

    return (
        true_value_for_attack,
        hard_true_value,
        event2value_attacker,
        optimizer_alpha,
        lr_scheduler,
        scaler,
        temperture_tau_scheduler,
    )


if __name__ == "__main__":
    main()
