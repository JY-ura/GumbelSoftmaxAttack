from functools import partial
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from torch.nn import functional as F

from attacks.ours.probability_space.probability_attack import ProbabilityFrameAttacker

from utils.loss_function import CrossEntropyLoss, L1Loss, MarginLoss, MSELoss

main_loss_dict = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MarginLoss": MarginLoss,
}

regular_loss_dict = {
    "MSELoss": MSELoss,
    "L1Loss": L1Loss,
}


class GradientAttackFactory(nn.Module):
    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the GradientAttackFactory.

        Args:
            cfg (dict): Configuration dictionary.
            model (nn.Module): Model to be attacked.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.model = model

        if "dnn" in self.cfg["model"]["name"]:
            self.forward = self.forward_dnn
        else:
            self.forward = self.forward_snn

    def attack_one_step(self, epoch: int):
        """
        Performs one step of the attack.

        Args:
            epoch (int): Current epoch.
        """
        raise NotImplementedError

    def forward_dnn(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass for DNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits tensor.
        """
        logits_list = []
        for i in range(x.shape[0]):
            logits = self.model(x[i])
            logits_list.append(logits)
        return torch.stack(logits_list).mean(0)

    def forward_snn(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass for SNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits tensor.
        """
        return self.model(x).mean(0)

class GumbelAttacker(GradientAttackFactory):
    def __init__(
        self,
        cfg: dict,
        probability_attacker: nn.Module,
        target_label: torch.Tensor,
        orginal_value: torch.Tensor,
        optimizer_alpha: torch.optim.Optimizer,
        model: nn.Module,
        scaler,
        lr_scheduler,
    ) -> None:
        """
        Initializes the Attacker class.

        Args:
            cfg (dict): Configuration dictionary.
            probability_attacker (nn.Module): Probability attacker module.
            target_label (torch.Tensor): Target label tensor.
            orginal_value (torch.Tensor): Original value tensor.
            optimizer_alpha (torch.optim.Optimizer): Alpha optimizer.
            model (nn.Module): Model module.
            scaler: Scaler object.
            lr_scheduler: Learning rate scheduler object.
        """
        super().__init__(
            cfg=cfg,
            model=model,
        )
        self.target_label = target_label
        self.orginal_value = orginal_value
        self.probability_attacker = probability_attacker
        self.optimizer_alpha = optimizer_alpha
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.use_soft_logits = cfg["use_soft_logits"]
        self.main_loss = main_loss_dict[cfg["main_loss_name"]](
            istargeted=cfg["targeted"],
            target=self.target_label,
            sample_num=cfg["attack"]["sample_num"],
            num_class=cfg["dataset"]["num_class"],
        )

        self.regular_loss = regular_loss_dict[cfg["regular_loss_name"]](
            self.orginal_value,
            sample_num=cfg["attack"]["sample_num"],
            reduction=cfg["regular_loss_reduction"],
        )

    def attack_one_step(self, epoch: int) -> Tuple[dict, dict]:
        """
        Performs one step of the attack.

        Args:
            epoch (int): Current epoch for one image attacking.

        Returns:
            Tuple[dict, dict]: A tuple containing the metrics and results.
        """
        functional.reset_net(self.model)
        if epoch % self.cfg["gradient_accumulation_steps"] == 0:
            self.optimizer_alpha.zero_grad()
        # hard_event: (sanple_num, nnz) hard_frame: (T, sample_num, channel, framesize, framesize)
        (
            hard_frame,
            soft_frame,
            hard_event_values,
            soft_event_values,
        ) = self.probability_attacker()

        attack_frame = soft_frame if self.cfg["use_soft_event"] else hard_frame
        attack_value = (
            soft_event_values if self.cfg["use_soft_event"] else hard_event_values
        )

        # using auto mixed precision
        with torch.cuda.amp.autocast(self.cfg["use_amp"]):  # type:ignore
            logits = self.forward(attack_frame)  # except T,N,C,H,W
            main_loss = self.main_loss(logits)
            regular_loss = self.regular_loss(attack_value)
            total_loss = main_loss + self.cfg["kappa"] * regular_loss

            if self.cfg["gradient_accumulation_steps"] > 1:
                total_loss /= self.cfg["gradient_accumulation_steps"]
                main_loss /= self.cfg["gradient_accumulation_steps"]
                regular_loss /= self.cfg["gradient_accumulation_steps"]

        # using Gradient scaling
        if self.cfg["use_grad_scaling"]:
            # torch.autograd.set_detect_anomaly(True)
            self.scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.probability_attacker.alpha, max_norm=20, norm_type=2)  # type: ignore
            # torch.nn.utils.clip_grad_value_(attacker.alpha, 1000)

            if epoch % self.cfg["gradient_accumulation_steps"] == 0:
                self.scaler.step(self.optimizer_alpha)
                self.scaler.update()

        else:
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.probability_attacker.alpha, max_norm=20, norm_type=2)  # type: ignore
            # self.probability_attacker.alpha
            if epoch % self.cfg["gradient_accumulation_steps"] == 0:
                self.optimizer_alpha.step()
                self.lr_scheduler.step()

                with torch.no_grad():
                    for param in self.probability_attacker.parameters():
                        param.clamp_(
                            -self.cfg["attack"]["alpha_boundary"],
                            self.cfg["attack"]["alpha_boundary"],
                        )
                        # print(torch.max(param), torch.min(param))

                # wandb.log({f"lr": self.lr_scheduler._last_lr[0]}, step=epoch)

        # generate logits of hardframe for validation
        with torch.no_grad():
            functional.reset_net(self.model)
            if self.use_soft_logits:
                assert (
                    soft_frame is not None
                ), "soft_frame is None, please check setting for probability_attacker"
                logits = self.forward(soft_frame)
            else:
                logits = self.forward(hard_frame)

        results = {
            "hard_event_values": hard_event_values.detach(),
            "logits": logits.detach(),
            "attack_frame": attack_frame,
        }
        metrics = {
            "total_loss": total_loss * self.cfg["gradient_accumulation_steps"],
            "main_loss": main_loss * self.cfg["gradient_accumulation_steps"],
            "regular_loss": regular_loss * self.cfg["gradient_accumulation_steps"],
        }

        return metrics, results
