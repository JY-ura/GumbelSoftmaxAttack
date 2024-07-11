"""
    Loss functions for adversarial attacks.
        - CrossEntropyLoss
        - MarginLoss
        - MSELoss
        - L1Loss
"""

import torch
from spikingjelly.activation_based import functional
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self, istargeted: bool, target: torch.Tensor, sample_num: int, num_class: int
    ) -> None:
        super().__init__()
        self.istargeted = istargeted
        self.target = target.repeat(sample_num)

    def forward(self, input: torch.Tensor):
        """
        Computes the cross-entropy loss.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if self.istargeted:
            return nn.CrossEntropyLoss()(input, self.target)
        else:
            return -nn.CrossEntropyLoss()(input, self.target)


class MarginLoss(nn.Module):
    """
    MarginLoss is a custom loss function used for adversarial attacks.

    Args:
        istargeted (bool): Whether the attack is targeted or not.
        target (torch.Tensor): The target tensor for the attack.
        sample_num (int): The number of samples.
        num_class (int): The number of classes.
    """

    def __init__(
        self, istargeted: bool, target: torch.Tensor, sample_num: int, num_class: int
    ) -> None:
        super().__init__()
        self.istargeted = istargeted
        target_onehot = functional.redundant_one_hot(target.unsqueeze(0), num_class, 1)
        self.target = target_onehot.repeat((sample_num, 1))

    def forward(self, input: torch.Tensor):
        target_score, _ = torch.max(input * self.target, axis=-1)  # type:ignore
        non_target_score, _ = torch.max(
            input * (1 - 1000 * self.target), axis=-1
        )  # type:ignore
        target_score = torch.clamp(target_score, min=1e-10)
        non_target_score = torch.clamp(non_target_score, min=1e-10)
        if self.istargeted:
            loss = torch.maximum(
                torch.tensor(0.0),
                torch.log(non_target_score + 1e-6) - torch.log(target_score + 1e-6),
            ).reshape(-1)
        else:
            loss = torch.maximum(
                torch.tensor(0.0),
                torch.log(target_score + 1e-6) - torch.log(non_target_score + 1e-6),
            ).reshape(-1)
        return torch.mean(loss)


class MSELoss(nn.Module):
    """
    Custom module for calculating the Mean Squared Error (MSE) loss between
    the original value and the adversarial image.

    Args:
        orginal_value (torch.Tensor): The original value to compare against.
        sample_num (int): The number of samples.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Default is "mean".

    Returns:
        torch.Tensor: The MSE loss between the original value and the adversarial image.
    """

    def __init__(
        self, orginal_value: torch.Tensor, sample_num, reduction="mean"
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.orginal_value = (
            orginal_value.repeat((sample_num, 1, 1))
            if orginal_value.ndim == 3
            else orginal_value.repeat((sample_num, 1))
        ).float()

    def forward(self, adv_image: torch.Tensor):
        return nn.MSELoss(reduction=self.reduction)(self.orginal_value, adv_image)


class L1Loss(nn.Module):
    def __init__(
        self, orginal_value: torch.Tensor, sample_num: int, reduction="mean"
    ) -> None:
        """
        L1 loss function.

        Args:
            orginal_value (torch.Tensor): The original value to compare against.
            sample_num (int): The number of samples.
            reduction (str, optional): The reduction method. Defaults to "mean".
        """
        super().__init__()
        self.reduction = reduction
        self.orginal_value = (
            orginal_value.repeat((sample_num, 1, 1))
            if orginal_value.ndim == 3
            else orginal_value.repeat((sample_num, 1))
        ).float()

    def forward(self, adv_image: torch.Tensor):
        """
        Forward pass of the L1 loss function.

        Args:
            adv_image (torch.Tensor): The adversarial image.

        Returns:
            torch.Tensor: The L1 loss between the original value and the adversarial image.
        """
        return nn.L1Loss(reduction=self.reduction)(self.orginal_value, adv_image)
