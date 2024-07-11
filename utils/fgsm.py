from typing import Tuple

import numpy as np
import torch

from .general_utils import get_one_target_label, sorted_indices_and_values


def init_value(event, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initializes alpha tensor from event dictionary.

    Args:
        event (npz): A dictionary containing 'p', 't', 'x', and 'y' keys.

    Returns:
        alpha (torch.Tensor): A tensor of shape (num_events, 3) containing the alpha values.
        indices (torch.Tensor): A tensor of shape (3, num_events) containing the indices of the event.
    """
    p = event["p"].astype(np.int32)
    p[p == 0] = -1
    values = torch.tensor(p.astype(np.float32), dtype=torch.float32)
    indices = np.array([event["t"], event["x"], event["y"]]).astype(np.int32)

    indices = torch.tensor(indices, dtype=torch.int32)
    return values, indices


def _get_attack_indices_values(
    event, device, auxiliary_event_dict, target_label, **kwarg
):
    """
    Get the attack indices and values for the Fast Gradient Sign Method (FGSM) attack.

    Args:
        event (torch.Tensor): The original event.
        device (torch.device): The device to perform the attack on.
        auxiliary_event_dict (dict): A dictionary containing target events for each label.
        target_label (torch.Tensor): The target label for the attack.
        **kwarg: Additional keyword arguments.

    Returns:
        torch.Tensor: The combined attack values.
        torch.Tensor: The combined attack indices.
    """
    orginal_values_, orginal_indices = init_value(event)
    target_events = auxiliary_event_dict.get(target_label.item())
    assert target_events is not None, "target label not found"
    # choose one of the target event
    random_index = np.random.choice(len(target_events))
    target_event = target_events[random_index]
    target_value, target_indices = init_value(target_event)

    vstack_indices = np.vstack([orginal_indices.T.numpy(), target_indices.T.numpy()])
    # vstack_indices_unique = np.unique(vstack_indices, axis=0)

    target_value[:] = 0.0

    orginal_value = torch.cat([orginal_values_, target_value], dim=0)

    # combine_value = orginal_value[0 : vstack_indices_unique.shape[0]]
    combine_indices = torch.tensor(vstack_indices.T)

    combine_indices, combine_value = sorted_indices_and_values(
        combine_indices, orginal_value
    )

    return combine_value.to(device), combine_indices.to(device)


def get_attack_indices_values(config: dict):
    """
    Get the attack indices and values for the Fast Gradient Sign Method (FGSM) attack.

    Args:
        config (dict): A dictionary containing the configuration parameters for the attack.

    Returns:
        torch.Tensor: The attack values as a tensor, with shape (1, num_features).
        torch.Tensor: The attack indices as a tensor, with shape (1, num_features).
    """
    while True:
        try:
            values, indices = _get_attack_indices_values(**config)
        except AssertionError as e:
            print("trying to change the target label")
            config["target_label"] = get_one_target_label(
                config["true_label"],
                config["device"],
                num_class=config["num_class"],
            )
        else:
            return values.unsqueeze(0), indices.unsqueeze(0)
