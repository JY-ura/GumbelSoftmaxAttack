from typing import Optional, Tuple

import torch
from torch import nn

from ..functional import HardDiffArgmax, SoftDiffArgmax


class TempretureSoftmax(nn.Module):
    """
    A module that applies temperature softmax to input probabilities.

    Args:
        tau (float): The temperature parameter for softmax. Default is 20.0.
        use_soft (bool): Whether to use soft argmax. Default is True.
    """

    def __init__(
        self, tau: float = 20.0, use_soft: bool = True, *args, **kwargs
    ) -> None:
        super().__init__()
        self.tau = tau
        self.sample_num = kwargs["sample_num"]
        self.use_soft = use_soft
        self.hard_argmax = HardDiffArgmax()
        self.soft_argmax = SoftDiffArgmax()
        assert self.sample_num == 1

    def forward(
        self, alpha, indices
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        indices = indices.unsqueeze(0).repeat_interleave(self.sample_num, dim=0)
        alpha = alpha.unsqueeze(0).repeat_interleave(self.sample_num, dim=0)
        tempretured_softmax = torch.softmax(alpha / self.tau, dim=-1)
        hard_event: torch.Tensor = self.hard_argmax.apply(
            tempretured_softmax
        )  # type:ignore
        if self.use_soft:
            soft_event = self.soft_argmax(tempretured_softmax)
            hard_event.detach_()
        else:
            soft_event = None
        return hard_event, soft_event, indices  # type:ignore
