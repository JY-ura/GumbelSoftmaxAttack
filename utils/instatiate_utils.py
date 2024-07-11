from typing import Tuple

import torch
from spikingjelly.datasets import cifar10_dvs, dvs128_gesture, n_mnist
from torch.utils.data import DataLoader, Subset, default_collate, random_split
from torchvision.datasets import VisionDataset

from attacks.ours.probability_space.event_generator import (
    GumbelSoftmaxTorch,
    TempretureSoftmax,
)
from utils.data_augment import DataAugment
from utils.general_utils import replace_all_batch_norm_modules_, stop_model_grad_
from utils.modified_model import get_model as _get_model

from .general_utils import log_grad

datasets_dict = {
    "cifar10-dvs": cifar10_dvs.CIFAR10DVS,
    "gesture-dvs": dvs128_gesture.DVS128Gesture,
    "nmnist": n_mnist.NMNIST,
}

generator_dict = {
    "torch": GumbelSoftmaxTorch,
    "tempreture": TempretureSoftmax,
}


def get_model(cfg: dict, device: torch.device):
    """Loads a pre-trained model based on the provided configuration. Also add hooks to watch the gradient if debug is True.

    Args:
    - cfg (dict): A configuration dictionary containing model-related settings.
    - device (torch.device): The device (e.g., CPU or GPU) on which the model should be loaded.

    Returns:
    - torch.nn.Module: The loaded pre-trained model.
    """
    model_config = cfg["model"].copy()
    model_path = model_config["model_path"]
    del model_config["model_path"]

    model = _get_model(
        is_train=False,
        **model_config,
    )

    if model_path is not None:
        if "pth" in model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path, map_location=device)["model"])

    model = replace_all_batch_norm_modules_(model)
    stop_model_grad_(model)
    model = model.to(device)

    if cfg["debug"]:
        for name, layer in model.named_modules():
            layer.register_full_backward_hook(log_grad(name))

    return model


def get_dataloaders(
    name: str,
    path: str,
    frame_number: int,
    batch_size: int,
    data_type: str,
    transform: dict,
    split_by="number",
    data_aug: bool = True,
    seed: int = 0,
    binary_frame: bool = True,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Subset]:
    """
    Get train, validation, and test data loaders for a given dataset.

    Args:
        name (str): Name of the dataset.
        path (str): Root path of the dataset.
        frame_number (int): Number of frames.
        batch_size (int): Batch size.
        data_type (str): Type of data.
        transform (dict): Data transformation dictionary.
        split_by (str, optional): Split method. Defaults to "number".
        data_aug (bool, optional): Whether to apply data augmentation. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        binary_frame (bool, optional): Whether the frames are binary. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[DataLoader, DataLoader, Subset]: Train, validation, and test data loaders.
    """
    train_augment = DataAugment(transform=transform, is_train=True)
    val_augment = DataAugment(transform=transform, is_train=False)

    # Load dataset
    if "cifar" in name:
        data: VisionDataset = datasets_dict[name](
            root=path,
            data_type=data_type,
            frames_number=frame_number,
            split_by=split_by,
        )
        train_data, test_data, val_data = random_split(
            data, [0.8, 0.1, 0.1], torch.Generator().manual_seed(seed)
        )  # type: ignore
    elif name in ["gesture-dvs", "nmnist"]:
        data: VisionDataset = datasets_dict[name](
            root=path,
            data_type=data_type,
            frames_number=frame_number,
            split_by=split_by,
            train=True,
        )
        test_data: VisionDataset = datasets_dict[name](
            root=path,
            data_type=data_type,
            frames_number=frame_number,
            split_by=split_by,
            train=False,
        )
        train_data, val_data = random_split(
            data, [0.9, 0.1], torch.Generator().manual_seed(seed)
        )
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    if data_aug:
        train_data.transform = train_augment  # type: ignore
        test_data.transform = val_augment  # type: ignore
        val_data.transform = val_augment  # type: ignore

    # Create DataLoader instances for train, validation, and test data
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        collate_fn=_collate_fn if not binary_frame else _collate_clamp_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        collate_fn=_collate_fn if not binary_frame else _collate_clamp_fn,
        drop_last=False,
    )
    # test_loader = DataLoader(
    #     dataset=test_data, batch_size=pics, pin_memory=True, num_workers=4, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_data  # type:ignore


def _collate_fn(batch):
    batch = default_collate(batch)
    imgs, labels = batch
    return (imgs.transpose(0, 1), labels)


def _collate_clamp_fn(batch):
    batch = default_collate(batch)
    imgs, labels = batch
    imgs = imgs.clamp(0, 1)  # clamp the input to [0, 1]
    return (imgs.transpose(0, 1), labels)


def get_event_generator(cfg: dict):
    """
    Get the event generator based on the configuration.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        generator: The event generator object.
    """
    name = cfg["attack"]["name"]
    generator_arguments = {
        "tau": cfg["attack"]["max_tau"],
        "sample_num": cfg["attack"]["sample_num"],
        "use_soft_event": cfg["use_soft_event"],
    }
    return generator_dict[name](**generator_arguments)
