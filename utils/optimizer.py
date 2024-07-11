import math
from functools import partial

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from .loss_function import CrossEntropyLoss, L1Loss, MarginLoss, MSELoss

main_loss_dict = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MarginLoss": MarginLoss,
}

regular_loss_dict = {
    "MSELoss": MSELoss,
    "L1Loss": L1Loss,
}

def get_loss(cfg, logits, target_label, origin_image, adv_image):
    """
    Calculate the total loss for adversarial attacks.

    Args:
        cfg (dict): Configuration dictionary containing parameters for the loss functions.
        logits (torch.Tensor): Model's output logits.
        target_label (torch.Tensor): Target label for the attack.
        origin_image (torch.Tensor): Original image tensor.
        adv_image (torch.Tensor): Adversarial image tensor.

    Returns:
        tuple: Total loss, main loss, and regular loss.
    """
    main_loss_function = main_loss_dict[cfg["main_loss_name"]](
        istargeted=cfg["targeted"],
        target=target_label,
        sample_num=cfg["attack"]["sample_num"],
        num_class=cfg["dataset"]["num_class"],
    )
    regular_loss_function = regular_loss_dict[cfg["regular_loss_name"]](
        reduction=cfg["regular_loss_reduction"], orginal_value=origin_image
    )
    main_loss = main_loss_function(logits)
    regular_loss = regular_loss_function(adv_image)
    # regular_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    return main_loss + cfg["kappa"] * regular_loss, main_loss, regular_loss


optimizer_dict = {"sgd": SGD, "adam": Adam}


def get_optimizer(params, **kwargs):
    """This function retrieves an optimizer for training a machine learning model based on the specified optimizer name and parameters.
    It uses a predefined dictionary (optimizer_dict) to map optimizer names to their corresponding optimizer classes.

    Args:
        params (generator): The parameters to be optimized.

    Returns:
        An instance of the selected optimizer class with the provided parameters.
    """
    name = kwargs["name"]
    del kwargs["name"]
    return optimizer_dict[name](params=params, **kwargs)


lr_fun_dict = {
    "CosALR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
}


def get_lr_scheduler(optimizer, **kwargs):
    name = kwargs["name"]
    del kwargs["name"]
    return lr_fun_dict[name](optimizer, **kwargs)


class CosineAnnealing:
    """
    Cosine annealing function that returns a value between initial_value and final_value based on the number of iterations.

    Parameters:
    - iters (int): The current iteration number.
    - initial_value (float): The initial value.
    - final_value (float): The final value.
    - decay_step (int): The number of iterations after which the value remains constant.

    Returns:
    - float: The annealed value between initial_value and final_value.
    """

    def __init__(
        self, optimizer, initial_value: float, final_value: float, decay_step: int
    ):
        self.optimizer = optimizer
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_step = decay_step

    def step(self, iters: int):
        if iters >= self.decay_step:
            value = self.initial_value
        else:
            value = self.initial_value + 0.5 * (
                self.final_value - self.initial_value
            ) * (1 + math.cos(math.pi * iters / self.decay_step))

        self.optimizer.tau = value
