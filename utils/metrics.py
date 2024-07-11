import datetime
import os
from dataclasses import dataclass, field
from pprint import pprint
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
from spikingjelly.datasets import play_frame, save_frames_to_npz_and_print
from torchvision import transforms

import wandb
from wandb import Table


@dataclass
class TotalMetrics:
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
        wandb.log(
            {"global_step": step, f"total_loss/total_loss_{image_id}": self.total_loss}
        )
        wandb.log(
            {
                "global_step": step,
                f"regular_loss/regular_loss_{image_id}": self.regular_loss,
            }
        )
        wandb.log(
            {"global_step": step, f"main_loss/main_loss_{image_id}": self.main_loss}
        )

    def reset(self):
        """
        Resets the metrics to zero.

        Returns:
        None
        """
        self.total_loss = 0
        self.regular_loss = 0
        self.main_loss = 0


@dataclass
class Metrics:
    finalize_metrics_dict: dict = field(default_factory=dict)
    event_l0_list: list = field(default_factory=list)
    event_l1_list: list = field(default_factory=list)
    event_l2_list: list = field(default_factory=list)
    frame_l0_list: list = field(default_factory=list)
    frame_l1_list: list = field(default_factory=list)
    frame_l2_list: list = field(default_factory=list)
    event_loss_num_list: list = field(default_factory=list)
    event_lost_ratio_list: list = field(default_factory=list)
    event_lost_ratio_ori_list: list = field(default_factory=list)
    perturbate_ratio_list: list = field(default_factory=list)
    average_num_event_list: list = field(default_factory=list)
    total_iter_success_list: list = field(default_factory=list)
    event_number_orginal_list: list = field(default_factory=list)
    event_number_adv_list: list = field(default_factory=list)
    total_num_success: int = 0
    total_num_correct: int = 0
    table = Table(
        columns=[
            "image_id",
            "main_loss",
            "regular_loss",
            "regular_loss_param",
            "perturbate_ratio",
            "iteration",
        ]
    )

    def update_metrics(
        self,
        cfg: dict,
        preds: torch.Tensor,
        target_label: torch.Tensor,
        results: dict,
        true_value: torch.Tensor,
        i: int,
        one_step_metrics: dict,
        true_frame: torch.Tensor,
        true_label: int,
        **kwargs,
    ):
        """update the metrics.

        Args:
            cfg (dict): config dict
            preds (torch.Tensor): the output of the model.
            target_label (torch.Tensor): the target label of the victim sample.
            results (dict): a dict of the successful sample attack results.
            true_value (torch.Tensor): the true value of the victim sample.
            i (int): the index of the victim sample.
            one_step_metrics (dict): one step metrics, including main_loss, regular_loss, total_loss.
            true_frame (torch.Tensor): the attack frame.
            true_label (int): the true label of the victim sample.
        """
        # mask
        mask = get_mask(preds, target_label, cfg["targeted"])
        adv_indices = results["hard_event_indices"].cpu()
        adv_values = results["hard_event_values"].detach()
        adv_frame = results["attack_frame"].cpu().detach()

        if cfg["is_save_img"]:
            save_image(
                true_frame=true_frame,
                adv_frame=adv_frame[:, mask, :],
                true_label=true_label,
                adv_label=int(preds[mask][0].item()),
                cfg=cfg,
                index=i,
            )

        # get the number of events lost
        adv_sparsity = get_num_of_event_lost(adv_values).to(preds.device)
        adv_sparsity_preds = adv_sparsity[mask]
        true_sample_sparsity = get_num_of_event_lost(true_value.unsqueeze(0)).to(
            preds.device
        )
        event_loss_num = torch.abs(
            torch.mean(true_sample_sparsity.float())
            - torch.mean(adv_sparsity_preds.float())
        )
        event_lost_ratio_adv = torch.mean(event_loss_num) / adv_values.shape[1]
        event_lost_ratio_ori = torch.mean(event_loss_num) / true_sample_sparsity
        average_num_event = torch.tensor(
            adv_values.shape[1], dtype=torch.float
        )  # total number of event4

        self.event_number_adv_list.append(adv_sparsity_preds.float().mean())

        # get l2 norm
        # adv_l2 = get_lp(adv_indices, adv_values, true_value, p=2)[mask]

        adv_l2 = torch.mean((adv_values - true_value) ** 2)
        # get l1 norm
        adv_l1 = get_lp(adv_indices, adv_values, true_value, p=1)[mask]
        # get l0 norm
        adv_l0 = get_l0(adv_indices, adv_values, true_value)[mask]
        perturbate_ratio = torch.mean(adv_l0) / torch.tensor(adv_values.shape[1])

        # frame metrics
        frame_l2 = torch.mean((adv_frame[:, mask, :] - true_frame) ** 2)
        # frame_l1 = torch.norm(adv_frame[:, mask, :] - true_frame, p=1)
        frame_l0 = torch.sum(
            (adv_frame[:, mask, :] - true_frame) != 0,
        ).float()

        # set metrics after evaluating
        self._update_metrics(
            cfg,
            adv_l0,
            adv_l1,
            adv_l2,
            frame_l2,
            # frame_l1,
            frame_l0,
            event_loss_num,
            event_lost_ratio_adv,
            average_num_event,
            perturbate_ratio,
            kwargs["iters"],
            i,
            one_step_metrics,
            event_lost_ratio_ori,
        )

    def _update_metrics(
        self,
        cfg,
        adv_l0,
        adv_l1,
        adv_l2,
        frame_l2,
        # frame_l1,
        frame_l0,
        event_loss_num,
        event_lost_ratio_adv,
        average_num_event,
        perturbate_ratio,
        iters,
        i,
        one_step_metrics,
        event_lost_ratio_ori,
    ):
        """
        Set metrics for an attack.

        Args:
            cfg (dict): The configuration dictionary.
            adv_l0 (torch.Tensor): L0 norm for each event in the adversarial attack.
            adv_l1 (torch.Tensor): L1 norm for each event in the adversarial attack.
            adv_l2 (torch.Tensor): L2 norm for each event in the adversarial attack.
            frame_l0 (torch.Tensor): L0 norm for each frame in the adversarial attack.
            frame_l1 (torch.Tensor): L1 norm for each frame in the adversarial attack.
            frame_l2 (torch.Tensor): L2 norm for each frame in the adversarial attack.
            event_loss_num (torch.Tensor): Number of events lost.
            event_lost_ratio (torch.Tensor): Ratio of events lost.
            average_num_event (torch.Tensor): Average number of events.
            perturbate_ratio (torch.Tensor): Perturbation ratio.
            iters (int): Number of iterations.
            i (int): the index of the victim sample.
            one_step_metrics (dict): one step metrics, including main_loss, regular_loss, total_loss.

        Returns:
            None
        """
        self.event_l0_list.append(torch.mean(adv_l0))
        self.event_l2_list.append(torch.mean(adv_l2))
        self.event_l1_list.append(torch.mean(adv_l1))
        self.frame_l0_list.append(torch.mean(frame_l0))
        self.frame_l2_list.append(torch.mean(frame_l2))
        # self.frame_l1_list.append(torch.mean(frame_l1))
        self.event_loss_num_list.append(event_loss_num)
        self.event_lost_ratio_list.append(event_lost_ratio_adv)
        self.event_lost_ratio_ori_list.append(event_lost_ratio_ori)
        self.average_num_event_list.append(average_num_event)
        self.perturbate_ratio_list.append(perturbate_ratio)
        self.total_iter_success_list.append(iters)
        self.total_num_success += 1
        self.table.add_data(
            i,
            f'{one_step_metrics["main_loss"].item():.4f}',
            f'{one_step_metrics["regular_loss"].item():.4f}',
            cfg["kappa"],
            f"{perturbate_ratio.item()*100:.2f}%",
            iters,
        )

    def finalize_metrics(self, start: float, end: float, cfg):
        """
        Finalizes the metrics by calculating various statistics and logging them using wandb.

        Args:
            start (float): The start time of the metric calculation.
            end (float): The end time of the metric calculation.
            cfg (dict): The configuration dictionary.

        Returns:
            None
        """
        num_pic = cfg["num_pic"]
        during_time = end - start

        if self.total_num_success == 0:
            self.finalize_metrics_dict = {
                "num_pic": num_pic,
                "time": f"{during_time/60.0:.2f} minutes({during_time/3600.0:.2f} hours).",
                "model_acc": f"{self.total_num_correct / num_pic * 100:.2f}%",
                "ASR": f"0.00%",
            }
            wandb.log(self.finalize_metrics_dict)
        else:
            event_lost_number = torch.stack(self.event_loss_num_list)
            event_lost_ratio_adv = torch.stack(self.event_lost_ratio_list)
            event_lost_ratio_ori = torch.stack(self.event_lost_ratio_ori_list)
            event_l2 = torch.stack(self.event_l2_list)
            event_l1 = torch.stack(self.event_l1_list)

            event_num_adv = torch.stack(self.event_number_adv_list)
            event_num_orgin = torch.tensor(self.event_number_orginal_list)
            event_l0 = torch.stack(self.event_l0_list)
            frame_l2 = torch.stack(self.frame_l2_list)
            # frame_l1 = torch.stack(self.frame_l1_list)
            frame_l0 = torch.stack(self.frame_l0_list)
            perturbate_ratio = torch.stack(self.perturbate_ratio_list)
            avg_num_event = torch.stack(self.average_num_event_list)
            asr = self.total_num_success / self.total_num_correct * 100

            self.finalize_metrics_dict = {
                # "num_pic": num_pic,
                "time": f"{during_time/60.0:.2f} minutes({during_time/3600.0:.2f} hours).",
                "model_acc": f"{self.total_num_correct / num_pic * 100:.2f}%",
                "ASR": f"{asr:.2f}%",
                # "success_samples_num": self.total_num_success,
                "iteration_max": f"{max(self.total_iter_success_list):.2f}",
                "iteration_mean": f"{mean(self.total_iter_success_list):.2f}",
                "event_l2 mean:": f"{torch.mean(event_l2).item():.4f} and std: {torch.std(event_l2).item():.4f}",
                # "event_l1 mean": f"{torch.mean(event_l1).item():.4f} and std: {torch.std(event_l1).item():.4f}",
                # "event_l0 mean": f"{torch.mean(event_l0).item():.4f} and std: {torch.std(event_l0).item():.4f}",
                "frame_l2 mean": f"{torch.mean(frame_l2).item():.4f} and std: {torch.std(frame_l2).item():.4f}",
                # "frame_l1 mean": f"{torch.mean(frame_l1).item():.4f} and std: {torch.std(frame_l1).item():.4f}",
                # "frame_l0 mean": f"{torch.mean(frame_l0).item():.4f} and std: {torch.std(frame_l0).item():.4f}",
                "avg_num_event": f"{torch.mean(avg_num_event).item():.2f} and std: {torch.std(avg_num_event).item():.2f}",
                "event_num_adv": f"{torch.mean(event_num_adv.float()):.2f} and std: {torch.std(event_num_adv.float()):.2f}",
                "event_num_orgin": f"{torch.mean(event_num_orgin.float()):.2f} and std: {torch.std(event_num_orgin.float()):.2f}",
                "event_num_all_space": f"{cfg['dataset']['frame_size']*cfg['dataset']['frame_size']*torch.mean(event_num_orgin.float()):.2f} \
                        and std: {cfg['dataset']['frame_size']*cfg['dataset']['frame_size']*torch.std(event_num_orgin.float()):.2f}",
                # "perturbate_ratio": f"{torch.mean(perturbate_ratio).item()*100:.4f} and std: {torch.std(perturbate_ratio).item()*100:.2f}",
                "event_change_number": f"{torch.mean(event_lost_number).item():.2f} and std: {torch.std(event_lost_number).item():.2f}",
                "event_change_ratio/adv": f"{torch.mean(event_lost_ratio_adv).item():.2f} and std: {torch.std(event_lost_ratio_adv).item():.2f}",
                "event_change_ratio/ori": f"{torch.mean(event_lost_ratio_ori).item():.2f} and std: {torch.std(event_lost_ratio_ori).item():.2f}",
                "ziteration_list": self.total_iter_success_list,
            }
            wandb.log(self.finalize_metrics_dict)
            # wandb.log({f'{cfg["name"]}_table': self.table})
        pprint(self.finalize_metrics_dict)


def upload_metrics(step: int, name: str, metrics: dict, image_id: int, attacker):
    """
    Uploads the metrics to the WandB platform.

    Args:
        step (int): The step number.
        name (str): The name of the metrics.
        metrics (dict): The metrics to be logged.
        image_id (int): The ID of the image.

    Returns:
        None
    """
    wandb.log(
        {
            "global_step": step,
            f"alpha_norm/alpha.norm_{image_id}": attacker.alpha.norm().item(),
        }
    )
    wandb.log(
        {"global_step": step, f"alpha_grad/alpha.grad_{image_id}": attacker.alpha.grad}
    )
    for metric_name, metric_value in metrics.items():
        wandb.log(
            {
                "global_step": step,
                f"{metric_name}/{metric_name}_{image_id}": metric_value,
            }
        )


@torch.no_grad()
def get_num_of_event_lost(event):
    """
    Calculates the sparsity of a set of events.

    The sparsity of an event is defined as the number of non-zero elements in the event.

    Args:
    - event (torch.Tensor): A 2D tensor representing a set of events where each row is an event.

    Returns:
    - torch.Tensor: A 1D tensor containing the sparsity values for each event in the input.
    """
    return torch.tensor([torch.sum(row != 0).item() for row in event])


@torch.no_grad()
def get_mask(preds: torch.Tensor, target_label: torch.Tensor, targeted: bool):
    if targeted:
        return preds == target_label
    else:
        return preds != target_label


@torch.no_grad()
def get_l0(adv_indices: torch.Tensor, adv_values: torch.Tensor, origin: torch.Tensor):
    """
    Calculate the L0 norm for each frame in the adversarial attack.

    Args:
        adv_indices (torch.Tensor): Tensor containing the indices of the adversarial attack.
        adv_values (torch.Tensor): Tensor containing the values of the adversarial attack.
        origin (torch.Tensor): Tensor containing the original values.

    Returns:
        torch.Tensor: Tensor containing the L0 norm for each frame in the adversarial attack.
    """
    frame_number = adv_indices.shape[0]
    l0_list = []
    for i in range(frame_number):
        l0_list.append(torch.sum((adv_values[i] - origin) != 0).float())
    return torch.stack(l0_list, dim=0)


@torch.no_grad()
def get_lp(indices, values, origin, p):
    """Calculates the Lp (p-norm) distance between a set of sparse events and an origin event.

    Args:
    - indices (torch.Tensor): A 2D tensor containing the indices of the sparse events. Each row corresponds to an event,
      and each row is a 2D tensor representing the indices of the non-zero elements of the event.
    - values (torch.Tensor): A 1D tensor containing the values associated with the non-zero elements of the events.
      It should have the same length as the number of non-zero elements in all events combined.
    - origin (torch.Tensor): The origin event for which the Lp distance is calculated.
    - p (float): The exponent in the Lp norm, where p >= 1.

    Returns:
    - torch.Tensor: A 1D tensor containing the Lp distances between each sparse event and the origin event.

    """

    sample_number = indices.shape[0]
    lp_list = []
    for i in range(sample_number):
        lp_list.append(torch.norm(values[i] - origin, p=p))
    return torch.stack(lp_list, dim=0)


def play_frame_split(x, save_gif_to: str):
    """
    Display frames from a tensor or numpy array and save them as a GIF.

    Args:
        x (torch.Tensor or np.ndarray): Input tensor or numpy array of shape [T, C, H, W].
        save_gif_to (str): Path to save the generated GIF file. If None, the frames will be displayed without saving.

    Returns:
        None
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]
    if save_gif_to is None:
        while True:
            for t in range(img_tensor.shape[0]):
                plt.imshow(to_img(img_tensor[t]))
                plt.pause(0.01)
    else:
        img_list = []
        for t in range(img_tensor.shape[0]):
            image = to_img(img_tensor[t])
            img_list.append(image)
            image.save(save_gif_to + f"_F{t}" + ".jpeg")
        img_list[0].save(
            save_gif_to + ".gif", save_all=True, append_images=img_list[1:], loop=0
        )
        # print(f"Save frames to [{save_gif_to}].")


def save_image(
    true_frame: torch.Tensor,
    adv_frame: torch.Tensor,
    true_label: int,
    adv_label: int,
    cfg: dict,
    index: int,
):
    """
    Save the true frame, adversarial frame, and the difference between them as images.

    Args:
        true_frame (torch.Tensor): The true frame.
        adv_frame (torch.Tensor): The adversarial frame. [T, num_samples, 2, H, W]
        true_label (int): The true label of the frame.
        adv_label (int): The adversarial label of the frame.
        cfg (dict): Configuration dictionary.
        index (int): Index of the frame.

    Returns:
        None
    """
    adv_frame = adv_frame[:, 0, :, :, :]  #  [T, 2, H, W]
    true_frame = true_frame[:, 0, :, :, :]  #  [T, 2, H, W]

    path = generate_path(cfg, index)

    name = f"img_true{true_label}_adv{adv_label}_"
    # play_frame_split(
    #     true_frame,
    #     adv_frame,
    #     path=path,
    #     name=name,
    # )
    play_frame_split(adv_frame, save_gif_to=path + str(index) + "adv_" + name)
    play_frame_split(true_frame, save_gif_to=path + str(index) + "org_" + name)


def generate_path(cfg, index: int) -> str:
    istargeted = "target" if cfg["targeted"] else "untarget"

    save_path = cfg["save_path"]
    if save_path == "":
        save_path = "./visual/"
    elif save_path[-1] != "/":
        save_path += "/"

    path = f"{save_path}{cfg['dataset']['name']}/{cfg['model']['name']}/{cfg['attack']['name']}/{istargeted}/{index}/"
    if not os.path.exists(path):
        os.makedirs(path)

    return path
