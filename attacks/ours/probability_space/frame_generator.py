from typing import Tuple

import numpy as np
import torch
from torch import nn, sparse
from torch.nn import functional as F


class FrameGenerator(nn.Module):
    def __init__(self, split_by: str, frame_number: int, frame_size: int) -> None:
        """
        Initializes a FrameGenerator object.

        Args:
            split_by (str): The method to split events into groups. Must be either 'time' or 'number'.
            frame_number (int): The number of frames to generate.
            frame_size (int): The size of a frame.
        """
        super().__init__()
        assert split_by in [
            "time",
            "number",
        ], "split_by must be either 'time' or 'number'"
        self.split_by = split_by
        self.frame_number = frame_number
        self.frame_size = frame_size

    def forward(
        self, event_values: torch.Tensor, event_indices: torch.Tensor, use_soft: bool
    ):
        """
        Splits events into groups based on the specified method and generates frames.

        Args:
            event_values (torch.Tensor): Batched event event_values with shape [sample_num, num_events].
            event_indices (torch.Tensor): Batched event_indices for events with shape [sample_num, num_events, ndim].
            use_soft (bool): Flag indicating whether to use soft splitting.

        Raises:
            ValueError: If split_by is not 'time' or 'number'.

        Returns:
            torch.Tensor: A tensor containing the generated frames.
        """
        if self.split_by == "time":
            raise NotImplementedError()
        elif self.split_by == "number":
            frames = self._split_events_by_number(event_values, event_indices, use_soft)
        else:
            raise ValueError(
                f"split_by must be either 'time' or 'number', got {self.split_by}"
            )

        return torch.transpose(frames, 0, 1)

    def _split_events_by_number(
        self, event_values: torch.Tensor, event_indices: torch.Tensor, use_soft: bool
    ) -> torch.Tensor:
        """
        Splits the events by number based on the given event values and indices.

        Args:
            event_values (torch.Tensor): The tensor containing the event values.
            event_indices (torch.Tensor): The tensor containing the event indices.
            use_soft (bool): A flag indicating whether to use soft splitting or hard splitting.

        Returns:
            torch.Tensor: The tensor containing the split events.
        """
        if use_soft:
            return self.soft_split_events_by_number(
                event_values.transpose(0, 1), event_indices
            )
        else:
            return self.hard_split_events_by_number(event_values.T, event_indices)

    def soft_split_events_by_number(
        self, event_values: torch.Tensor, event_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Splits events into groups based on the number of events and generates frames.

        Args:
            event_values (torch.Tensor): Batched event event_values with shape [num_events, sample_num, 2].
            event_indices (torch.Tensor): Batched event_indices for events with shape [sample_num, ndim, num_events].

        Returns:
            torch.Tensor: A tensor containing the generated frames.
        """
        time_window = event_values.shape[0] // self.frame_number
        frames = []
        for i in range(self.frame_number):
            start = i * time_window
            end = start + time_window
            sub_values = event_values[start:end, :, :]
            sub_indices = event_indices[:, :, start:end]
            shape = (
                event_indices.shape[0],
                time_window,
                self.frame_size,
                self.frame_size,
            )

            frame_channel0 = _integrate_events_to_frame(
                values=sub_values[:, :, 0], indices=sub_indices, shape=shape
            )
            frame_channel1 = _integrate_events_to_frame(
                values=sub_values[:, :, -1], indices=sub_indices, shape=shape
            )

            frame = torch.stack([frame_channel0, frame_channel1], dim=0)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = frames.to_dense().permute(-1, 0, 1, 3, 2)
        return frames

    def hard_split_events_by_number(
        self, event_values: torch.Tensor, event_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Splits events into groups based on the number of events and generates frames.

        Args:
            event_values (torch.Tensor): Batched event hard_values with shape [num_events, sample_num].
            event_indices (torch.Tensor): Batched indices for events with shape [sample_num, ndim, num_events].

        Returns:
            torch.Tensor: A tensor containing the generated frames.
        """
        # event_values.requires_grad = True
        time_window = event_values.shape[0] // self.frame_number
        frames = []
        for i in range(self.frame_number):
            start = i * time_window
            end = start + time_window
            group_values = event_values[start:end, :]
            group_indices = event_indices[:, :, start:end]
            # accumulate -1 and 1 separately into 2 channels
            pos_group_values = F.relu(group_values)
            neg_group_values = F.relu(-group_values)

            shape = (
                event_indices.shape[0],
                time_window,
                self.frame_size,
                self.frame_size,
            )
            pos_channel = _integrate_events_to_frame(
                pos_group_values, group_indices, shape
            )
            neg_channel = _integrate_events_to_frame(
                neg_group_values, group_indices, shape
            )
            frame = torch.stack([neg_channel, pos_channel], dim=0)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = frames.to_dense().permute(-1, 0, 1, 3, 2)
        return frames


def _integrate_events_to_frame(
    values: torch.Tensor, indices: torch.Tensor, shape: Tuple
) -> torch.Tensor:
    """
    Integrate events to form a frame.

    Args:
        values (torch.Tensor): Batched event values with shape [ num_events, sample_num].
        indices (torch.Tensor): Batched indices for events with shape [sample_num, ndim, num_events].
        shape (_type_): Shape of the output frame, with format [sample_num, T, X, Y].

    Returns:
        torch.Tensor: Integrated event frame along T dim with shape [sample_num, X, Y].
    """

    indices = indices[0, :, :]
    shape = (indices[0, -1].item() + 1,) + shape[2:] + (shape[0],)
    # create sparse tensor for integration
    event_frame = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape,  # making the batch into dense dimension #type:ignore
        device=values.device,
    ).coalesce()
    # event_frame_dense = event_frame.to_dense()
    event_frame = torch.sparse.sum(event_frame, dim=0)

    return event_frame
