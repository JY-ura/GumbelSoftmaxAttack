from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

import wandb


def log_grad(name):
    def hook(model, input, output):
        wandb.log({name: output[0]})

    return hook


def get_device(index: int):
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")


def get_index_of_attack_sample(data, num_pic):
    return np.random.choice(len(data), num_pic, replace=False)

def get_true_values(alpha: torch.Tensor, use_soft_event: bool):
    """
    If you use soften value, leave the two channels -1 and 1 respectively;
    otherwise, keep one channel that contains -1 and 1.
    """
    soft_values = torch.stack(
        [alpha[:, 0].eq(1).int(), alpha[:, 2].eq(1).int()]
    ).float()
    soft_values = soft_values.T.unsqueeze(0)

    hard_values = torch.argmax(alpha, dim=-1).unsqueeze(-1) - 1
    hard_values = hard_values.unsqueeze(0)
    if use_soft_event:
        return hard_values[0].squeeze(-1), soft_values
    return hard_values[0].squeeze(-1), hard_values.squeeze(-1)


def get_onehot_label_batch(cfg: dict, target_label: torch.Tensor):
    """
    Generate a batch of one-hot encoded labels based on the target label.

    Args:
        cfg (dict): Configuration dictionary.
        target_label (torch.Tensor): Target label tensor with shape [num_correct].

    Returns:
        torch.Tensor: Batch of one-hot encoded labels with shape [sample_numbel, num_correct, num_class]
    """
    one_hot_label = target_label.unsqueeze(-1).repeat((1, cfg["attack"]["sample_num"]))
    return one_hot_label


def get_one_target_label(true_label: int, device: torch.device, num_class: int):
    """
    Prepare the target label for adversarial attack. It is for initial alpha, you know, target attacks need a target sample to generate the alpha.

    Args:
        cfg (dict): Configuration dictionary.
        true_label (int): The true label of the input sample.
        index (int): Index of the input sample.
        device (torch.device): Device to be used for computation.

    Returns:
        torch.Tensor: Batch of one-hot encoded target labels.
        torch.Tensor: Target label for the input sample.
    """
    target_label = torch.tensor(
        get_random_target(num_class, true_label),
        dtype=torch.int64,
        device=device,
    )
    return target_label


def get_target_label(cfg: dict, true_labels, device: torch.device):
    """
    Get the target labels for the attack.

    Args:
        cfg (dict): Configuration dictionary.
        true_labels: True labels of the input data.
        device (torch.device): Device to store the target labels.

    Returns:
        torch.Tensor: Target labels for the attack.
    """
    if not cfg["targeted"]:
        target_labels = torch.tensor(true_labels, dtype=torch.int64, device=device)
    else:
        _target_labels = []
        for true_label in true_labels:
            target_label = torch.tensor(
                get_random_target(cfg["dataset"]["num_class"], true_label),
                dtype=torch.int64,
                device=device,
            )
            _target_labels.append(target_label)
        target_labels = torch.tensor(_target_labels, dtype=torch.int64, device=device)
    # onehot_label_batch = get_onehot_label_batch(cfg, target_labels)
    print("true   labels:", true_labels)
    print("target labels:", target_labels)
    return target_labels


def pre_process(event, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """tranfer event data from NpzFile to torch.Tensor.

    Args:
        event (numpy.lib.npyio.NpzFile): containing event data, including 'p,' 't,' 'x,' and 'y' values.
        device (cuda): The device on which to place the resulting tensors.

    Returns:
        - A tuple of two tensors: (indices, values)
            indices (torch.Tensor): A tensor containing 't,' 'x,' and 'y' values on the specified device.
            values (torch.Tensor): A tensor containing the transformed 'p' array as floats on the specified device.
    """
    p = event["p"]
    p[p == 0] = -1
    values = torch.tensor(p.astype(np.int32), dtype=torch.int32, device=device)
    indices = torch.tensor(
        np.array([event["t"], event["x"], event["y"]]).astype(np.int32), device=device
    )
    return indices.unsqueeze(0), values.unsqueeze(0).float()


def get_random_target(total_indices: int, true_class: int) -> int:
    """Generates a pseudorandom targeted label different from the true class.

    Args:
    - index (int): An index(seed) for the random number generator.
    - total_indices (int): The total number of possible target labels.
    - true_class (int): The true class for which a targeted label is being generated.

    Returns:
    - int: A pseudorandom target label that is different from the true class.
    """
    target = true_class
    while target == true_class:
        target = np.random.randint(0, total_indices)
    return target


def is_success(
    preds: torch.Tensor, labels: torch.Tensor, targeted: bool = False
) -> torch.Tensor:
    """Check whether the attack succeeds.

    Args:
        preds (torch.Tensor): Predicted values.
        labels (torch.Tensor): Target label.
        targeted (bool): Whether the attack is targeted (default is False).

    Returns:
        torch.Tensor: True or False.
    """
    if targeted:
        return (preds == labels).any()
    else:
        return (preds != labels).any()


def remove_misclassification(cfg: dict, test_data, model, frame_processor, device):
    """
    Removes misclassified samples from the test data based on the given model's predictions.

    Args:
        cfg (dict): Configuration dictionary.
        test_data: Test data containing events and labels.
        model: Trained model used for prediction.
        frame_processor: Frame processor object.
        device: Device used for computation.

    Returns:
        Tuple containing the number of correctly classified samples, correct events, true labels,
        true values, true indices, and frame list.
    """
    index_of_attack_list = get_index_of_attack_sample(
        data=test_data, num_pic=cfg["num_pic"]
    )
    correct_events = []
    true_labels = []
    true_values = []
    true_indices = []
    frame_list = []
    print("Removing misclassified samples ……")
    for i, img_index in enumerate(index_of_attack_list):
        event, true_label = test_data[img_index]
        true_indice, values_1d = pre_process(event=event, device=device)
        unattack_frame = frame_processor.forward(
            event_indices=true_indice, event_values=values_1d, use_soft=False
        )
        if not has_correct_prediction(cfg, model, unattack_frame, true_label):
            # print(f"Image:{i} has wrong prediction, skip")
            continue
        correct_events.append(event)
        true_labels.append(true_label)
        true_values.append(values_1d)
        true_indices.append(true_indice)
        frame_list.append(unattack_frame.cpu())
    correct_num = len(true_labels)
    print(f"model accuracy: {correct_num/cfg['num_pic']*100:.2f}%")
    return (
        correct_num,
        correct_events,
        true_labels,
        true_values,
        true_indices,
        frame_list,
    )


def forward_dnn(
    x: torch.Tensor,
    model,
):
    """
    Forward pass through a DNN model.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ...)
        model: The DNN model to forward pass through

    Returns:
        torch.Tensor: Output logits tensor of shape (batch_size, ...)
    """
    logits_list = []
    for i in range(x.shape[0]):
        logits = model(x[i])
        logits_list.append(logits)
    return torch.stack(logits_list).mean(0)


def forward_snn(
    x: torch.Tensor,
    model,
):
    """
    Forward pass through the spiking neural network (SNN) model.

    Args:
        x (torch.Tensor): Input tensor.
        model: The SNN model.

    Returns:
        torch.Tensor: Output tensor after passing through the SNN model.
    """
    return model(x).mean(0)


def has_correct_prediction(cfg, model, unattack_frame, true_label):
    """
    Checks if the model's prediction matches the true label for a given unattacked frame.

    Args:
        cfg (dict): Configuration dictionary.
        model: The model to evaluate.
        unattack_frame: The unattacked frame to evaluate.
        true_label: The true label of the unattacked frame.

    Returns:
        bool: True if the model's prediction matches the true label, False otherwise.
    """
    with torch.no_grad():
        functional.reset_net(model)
        if "dnn" not in cfg["model"]["name"]:
            forward = forward_snn
        else:
            forward = forward_dnn
        pred = torch.argmax(forward(unattack_frame, model=model), dim=-1)
        torch.cuda.empty_cache()

    return pred == true_label


def generate_auxiliary_samples_for_each_sample(
    cfg, true_labels: list, correct_events: list
) -> dict:
    """
    Generates auxiliary samples for each true label.

    Args:
        cfg: The configuration object.
        true_labels (list): The list of true labels.
        correct_events (list): The list of correct events.

    Returns:
        dict: A dictionary where the keys are the true labels and the values are lists of corresponding correct events.
    """
    auxiliary_event = {}
    for i, key in enumerate(true_labels):
        if key in auxiliary_event:
            auxiliary_event[key].append(correct_events[i])
        else:
            auxiliary_event[key] = [correct_events[i]]
    return auxiliary_event


def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # base case
    _batch_norm_without_running_stats(root)

    for obj in root.modules():
        _batch_norm_without_running_stats(obj)
    return root


def _batch_norm_without_running_stats(module: nn.Module):
    if (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ):
        module.eval()


def stop_model_grad_(net: nn.Module):
    for params in net.parameters():
        params.requires_grad = False


def sorted_indices_and_values(indices, values):
    """sort indices and values based on the indices, it is for probability_attack.py.

    Args:
        indices (torch.Tensor): indices shape [3, num_events]
        values (torch.Tensor): values shape [num_events, 3]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: sorted indices and values
    """
    first_row_values = indices[0, :]
    index = torch.argsort(first_row_values)

    sorted_indices = torch.gather(
        indices, 1, index.unsqueeze(0).repeat(indices.shape[0], 1)
    )
    if values.dim() == 1:
        sorted_values = torch.gather(values, 0, index)
    elif values.dim() == 2:
        sorted_values = torch.gather(
            values, 0, index.unsqueeze(1).repeat(1, values.shape[1])
        )
    else:
        raise ValueError("values dim should be 1 or 2")
    return sorted_indices, sorted_values


def get_target_label_for_add_position(
    add_position_label_mode: str, target_label: torch.Tensor, num_class: int
):
    """
    Get the target label for adding a position.

    Args:
        add_position_label_mode (str): The mode for adding a position label. Possible values are "target" and "random_except_target".
        target_label (torch.Tensor): The target label tensor.
        num_class (int): The number of classes.

    Returns:
        torch.Tensor: The target label for adding a position.

    """
    if add_position_label_mode == "target":
        return target_label
    elif add_position_label_mode == "random_except_target":
        target_label = (
            target_label
            + torch.randint(
                low=1, high=num_class, size=(1,), device=target_label.device
            )
        ) % num_class
        return target_label.squeeze(0)
