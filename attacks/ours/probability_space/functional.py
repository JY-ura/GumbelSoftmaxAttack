from typing import Tuple

import torch
import torch.nn as nn
from torch import autograd


class HardDiffArgmax(autograd.Function):
    """
    Differentiable argmax function

    Args:
        x (torch.Tensor): Input tensor
        num_class (int): Number of classes

    Returns:
        torch.Tensor: Tensor containing the differentiable argmax values
    """

    @staticmethod
    def forward(ctx, x, num_class: int = 3):
        argmax_value = torch.argmax(x, dim=-1).float() - 1
        ctx.save_for_backward(argmax_value)
        ctx.num_class = num_class
        # soft_value, _ = torch.max(x, dim=-1)
        return argmax_value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        argmax_value = ctx.saved_tensors[0]
        num_class = ctx.num_class
        grad_mask = (
            torch.arange(3, device=argmax_value.device).unsqueeze(0).unsqueeze(0)
            == argmax_value.unsqueeze(-1) + 1
        ).int()
        grad_input = torch.stack([grad_output for _ in range(num_class)], dim=-1)
        grad_input = grad_input * grad_mask

        return grad_input, None


class SoftDiffArgmax(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, num_class: int = 3):
        """
        Applies the SoftDiffArgmax operation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            num_class (int): Number of classes. Default is 3.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, 2), where the first dimension
                          represents the batch size, the second dimension represents the sequence length, and
                          the third dimension represents the two selected elements from the input tensor.
        """
        x_0 = x[:, :, 0]
        x_1 = x[:, :, -1]
        x_remove_0 = torch.stack([x_0, x_1], dim=-1)
        return x_remove_0
