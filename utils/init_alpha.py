from typing import Tuple

import numpy as np
import torch

from utils.general_utils import get_one_target_label, sorted_indices_and_values


# no useful because the memory is not enough
def remove_repeat_indices(random_indices, origin_indices):
    filtered_random_indices = random_indices[
        ~np.any(np.all(random_indices[:, None] == origin_indices, axis=-1), axis=-1)
    ]
    return filtered_random_indices[0]


def _add_indices(origin_indices, init_event_num):

    init_time = np.random.randint(
        low=0,
        high=np.max(origin_indices[0]),  # type:ignore
        size=(init_event_num, 1),
    )
    init_x = np.random.randint(
        low=0,
        high=np.max(origin_indices[1]),  # type:ignore
        size=(init_event_num, 1),
    )
    init_y = np.random.randint(
        low=0,
        high=np.max(origin_indices[2]),  # type:ignore
        size=(init_event_num, 1),
    )
    random_indices = np.concatenate([init_time, init_x, init_y], axis=1)

    new_indices = np.concatenate([origin_indices.T, random_indices]).transpose()

    return new_indices


def add_indices_values(
    values: torch.Tensor,
    indices: np.ndarray,
    alpha: torch.Tensor,
    device: torch.device,
    add_ratio_events: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add additional indices and values to the given tensor.

    Args:
        values (torch.Tensor): The tensor of values.
        indices (np.ndarray): The tensor of indices.
        alpha (torch.Tensor): The tensor of alpha values.
        device (torch.device): The device to use for the new tensors.

    Returns:
        torch.Tensor: The updated tensor of alpha values.
        torch.Tensor: The updated tensor of indices.
    """
    event_num = values.shape[0]
    init_event_num = int(event_num * add_ratio_events)
    new_indices = _add_indices(indices, init_event_num)

    add_value = (
        torch.ones(size=(new_indices.shape[1] - event_num, 3), device=device) * 0.05
    )
    add_value[:, 1] = 0.9
    new_alpha = torch.cat([alpha, add_value], dim=0)
    new_indices = torch.tensor(new_indices, device=device)
    return new_alpha, new_indices


def init_alpha_from_events(
    events, device: torch.device, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initializes alpha tensor from events dictionary.

    Args:
        events (npz): A input containing 'p', 't', 'x', and 'y' keys.
        device (torch.device): The device to create the indices tensor on.

    Returns:
        alpha (torch.Tensor): A tensor of shape (num_events, 3) containing the alpha values.
        indices (torch.Tensor): A tensor of shape (3, num_events) containing the indices of the events.
    """
    p = events["p"]
    p[p == 0] = -1
    values = torch.tensor(p.astype(np.int32), dtype=torch.int32)
    indices = np.array([events["t"], events["x"], events["y"]]).astype(np.int32)

    # create alpha from values
    mask = torch.arange(3).unsqueeze(-1)
    alpha = ((values + 1) == mask).float().to(device).T

    indices = torch.tensor(indices, dtype=torch.int32)
    return alpha, indices


def init_alpha_from_frame(
    frame: torch.Tensor, device: torch.device, **kwargs
) -> torch.Tensor:
    """
    Initializes alpha tensor from binary evetns, alpha would only have 0 or 1 value.

    Args:
        frame (torch.Tensor): A tensor of shape (num_events, 3) containing the alpha values.
        device (torch.device): The device to create the indices tensor on.

    Returns:
        alpha (torch.Tensor): A tensor of shape (num_events, 3) containing the alpha values.
    """
    alpha_0 = (frame == 0).float()
    alpha_1 = (frame == 1).float()
    alpha = torch.stack([alpha_0, alpha_1], dim=-1).to(device)
    return alpha


def init_alpha_random_add_indices(
    events, device: torch.device, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initializes the alpha tensor and indices tensor for the probability attack. randomly add indices and values.

    Args:
        events (npz): The input events.
        device (torch.device): The device to be used for computation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the alpha tensor and indices tensor.
    """
    p = events["p"]
    p[p == 0] = -1
    values = torch.tensor(p.astype(np.int32), dtype=torch.int32)
    np_indices = np.array([events["t"], events["x"], events["y"]]).astype(np.int32)

    # create alpha from values
    mask = torch.arange(3).unsqueeze(-1)
    alpha = ((values + 1) == mask).float().to(device).T

    alpha, indices = add_indices_values(
        values, np_indices, alpha, device, kwargs["target_position_ratio"]
    )

    combine_indices, combine_alpha = sorted_indices_and_values(indices, alpha)
    return combine_alpha, combine_indices


def init_alpha_target(events, device, event_dict, target_label, **kwargs):
    """
    Initialize the alpha values according to the target sample.

    Args:
        events (npz): The input events.
        device (torch.device): The device to use for computation.
        event_dict (dict): A dictionary mapping labels to corresponding events.
        target_label (torch.Tensor): The target label for the attack.

    Returns:
        combine_alpha (torch.Tensor): The synthetic alpha values.[num_events, 1]
        combine_indices (torch.Tensor): The synthetic indices.[num_events, 3]
    """
    orginal_alpha_, orginal_indices = init_alpha_from_events(events, device)
    target_events = event_dict.get(target_label.item())
    assert target_events is not None, "target label not found"
    # choose one of the target events
    random_index = np.random.choice(len(target_events))
    target_event = target_events[random_index]
    target_alpha, target_indices = init_alpha_from_events(target_event, device)

    # target_alpha.shape : torch.size([218594, 3])
    # target_indices.shape : torch.size([3, 218594])
    number = int(target_alpha.shape[0] * kwargs["target_position_ratio"])
    random_choose_index = torch.randperm(target_alpha.size(0))
    target_alpha = target_alpha[random_choose_index[:number]]
    target_indices = target_indices[:, random_choose_index[:number]]

    vstack_indices = np.vstack([orginal_indices.T.numpy(), target_indices.T.numpy()])

    target_alpha[:, 0] = 0.05
    target_alpha[:, 1] = 0.9
    target_alpha[:, 2] = 0.05

    combine_alpha = torch.cat([orginal_alpha_, target_alpha], dim=0).to(device)
    combine_indices = torch.tensor(vstack_indices.T).to(device)

    combine_indices, combine_alpha = sorted_indices_and_values(
        combine_indices, combine_alpha
    )

    return combine_alpha, combine_indices


init_alpha_mode_dict = {
    "default": init_alpha_from_events,
    "random_add_indices": init_alpha_random_add_indices,
    "target": init_alpha_target,
}


def get_alpha(parameters: dict):
    """
    Retrieves the alpha value and indices based on the given parameters.

    Args:
        parameters (dict): A dictionary containing the necessary parameters.

    Returns:
        tuple: A tuple containing the alpha value and indices.
    """
    while True:
        try:
            alpha, indices = init_alpha_mode_dict[parameters["init_alpha_mode"]](
                **parameters
            )
        except AssertionError as e:
            print("trying to change the target label...")
            parameters["target_label"] = get_one_target_label(
                parameters["true_label"],
                parameters["device"],
                num_class=parameters["num_class"],
            )
        else:
            return alpha, indices
