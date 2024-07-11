"""
This module contains generators that generate spike event from probability space
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax

from ..functional import HardDiffArgmax, SoftDiffArgmax


class GumbelSoftmaxTorch(nn.Module):
    """
    GumbelSoftmaxTorch module applies the Gumbel-Softmax relaxation to a given input tensor.

    Args:
        tau (float): The temperature parameter for the Gumbel-Softmax relaxation. Default is 1.
        sample_num (int): The number of samples to generate using Gumbel-Softmax relaxation. Default is 1.
        use_soft_event (bool): Whether to use soft event or not. Default is True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the hard event tensor, soft event tensor, and indices tensor.
    """

    def __init__(
        self,
        tau: float = 1,
        sample_num: int = 1,
        use_soft_event: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.sample_num = sample_num
        self.use_soft_event = use_soft_event
        self.soft_argmax = SoftDiffArgmax()
        self.hard_argmax = HardDiffArgmax()

    def forward(
        self, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        alpha = alpha.unsqueeze(0).repeat_interleave(
            self.sample_num, dim=0
        )  # [sample_number, event_number, 3]
        soften_gumbel_3d = gumbel_softmax(alpha, tau=self.tau, hard=False)
        hard_event: torch.Tensor = self.hard_argmax.apply(
            soften_gumbel_3d
        )  # type:ignore
        if self.use_soft_event:
            soft_event = self.soft_argmax(soften_gumbel_3d)
            hard_event.detach_()
        else:
            soft_event = None
        return hard_event, soft_event  # type:ignore hard_event [sample_number, event_number]
