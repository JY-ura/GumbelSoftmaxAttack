"""
This module contains the implementation of the probability space attack.
By optimizing the parameter of the probability, we turn the discrete categorical attack into continuous.
"""

from typing import Callable, Optional

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.functional import gumbel_softmax

from attacks.ours.probability_space.event_generator.gumbel_torch import (
    GumbelSoftmaxTorch,
)
from utils.init_alpha import get_alpha

from .frame_generator import FrameGenerator


class ProbabilityAttacker(nn.Module):
    def __init__(
        self,
        attack_cfg: dict,
        alpha_dict: dict,
        event_generator: Optional[Callable] = None,
        frame_processor: Optional[Callable] = None,
    ) -> None:
        """
        ProbabilityAttacker class represents an attacker that performs probability-based attacks.

        Args:
            attack_cfg (dict): Configuration parameters for the attack.
            alpha_dict (dict): Dictionary containing alpha values.
            event_generator (Optional[Callable]): Callable object for generating events. Defaults to None.
            frame_processor (Optional[Callable]): Callable object for processing frames. Defaults to None.
        """
        super().__init__()
        self.sample_num = attack_cfg["sample_num"]
        self.lamda = attack_cfg["lamda"]
        self.tau = attack_cfg["max_tau"]
        self.use_soft_event = attack_cfg["use_soft_event"]

        alpha, event_indices = get_alpha(alpha_dict)
        self.alpha = nn.parameter.Parameter(data=alpha, requires_grad=True)
        self.event_indices = event_indices.unsqueeze(0).repeat_interleave(
            self.sample_num, dim=0
        )

        if event_generator is None:
            event_generator = GumbelSoftmaxTorch(
                tau=self.tau, sample_num=self.sample_num, use_soft=self.use_soft
            )

        if frame_processor is None:
            frame_processor = FrameGenerator(
                split_by="number", frame_number=16, frame_size=128
            )

        self.event_generator = event_generator
        self.frame_processor = frame_processor

    def forward(self):
        hard_values, soft_values = self.event_generator(self.alpha)
        # hard_values.retain_grad()

        if self.use_soft_event:
            soft_frame = self.frame_processor(
                soft_values, self.event_indices, use_soft=self.use_soft_event
            )
        else:
            soft_frame = None
        hard_frame = self.frame_processor(
            hard_values, self.event_indices, use_soft=False
        )
        return hard_frame, soft_frame, hard_values, soft_values


class SoftArgmax2d(Function):
    """
    SoftArgmax2d function performs a differentiable approximation of the argmax operation
    on a 2D input tensor.

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor containing the indices of the maximum values
                      along the specified dimension.

    Examples:
        >>> input = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> output = SoftArgmax2d.apply(input)
        >>> print(output)
        tensor([[2., 2., 2.], [2., 2., 2.]])
    """

    @staticmethod
    def forward(ctx, input):
        hard_value = torch.argmax(input, dim=-1).float()
        ctx.save_for_backward(hard_value)
        return hard_value

    @staticmethod
    def backward(ctx, grad_output):
        (hard_value,) = ctx.saved_tensors
        shape = hard_value.shape
        grad_mask = torch.arange(2, device=hard_value.device)
        for i in range(len(shape) - 1):
            grad_mask = grad_mask.unsqueeze(0)
        grad_mask = (grad_mask == hard_value.unsqueeze(-1)).float()
        grad_output = torch.stack([grad_output, grad_output], dim=-1)
        return grad_mask * grad_output


class ProbabilityFrameAttacker(nn.Module):
    def __init__(
        self, attack_cfg, binary_frame: torch.Tensor, device, **kwargs
    ) -> None:
        """
        ProbabilityFrameAttacker class for generating adversarial frames using probability space attack.

        Args:
            attack_cfg (dict): Configuration for the attack.
            binary_frame (torch.Tensor): Binary frame used for initializing alpha.
            device: Device on which the computation will be performed.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()
        self.sample_num = attack_cfg["sample_num"]
        self.lamda = attack_cfg["lamda"]
        self.tau = attack_cfg["max_tau"]
        self.use_soft_event = attack_cfg["use_soft_event"]

        self.alpha = nn.parameter.Parameter(
            data=init_alpha_from_frame(binary_frame, device=device), requires_grad=True
        )

    def forward(
        self,
    ):
        """
        Forward pass of the ProbabilityFrameAttacker.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the hard frame and soft frame.

        """
        alpha = self.alpha.unsqueeze(0).repeat_interleave(self.sample_num, dim=0)
        hard_frame = gumbel_softmax(alpha, tau=self.tau, hard=True, dim=-1)
        hard_frame = SoftArgmax2d.apply(hard_frame)
        soft_frame = gumbel_softmax(alpha, tau=self.tau, hard=False, dim=-1)
        indices = torch.arange(2, device=alpha.device)
        for _ in range(len(hard_frame.shape) - 1):
            indices = indices.unsqueeze(0)
        soft_frame = torch.sum(soft_frame * indices, dim=-1)

        if self.use_soft_event:
            hard_frame.detach_()
        else:
            soft_frame.detach_()

        # shape [sample_num, H, W, 2]
        return hard_frame.transpose(0, 1), soft_frame.transpose(0, 1)
