from typing import Any

import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import InterpolationMode


def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class DataAugment:
    """This class defines a data augmentation pipeline for image data,
    which includes the following transformations: AutoAugment, RandomErasing, and Normalization.
    """

    def __init__(self, transform, is_train=False, use_v2=False) -> None:
        """Initialize the DataAugment class with the given parameters.
        
        Args:
            transform (dict): Configuration for the transformations.
            is_train (bool): Flag to indicate if the transformations are for training.
            use_v2 (bool): Flag to use version 2 of the transformation module.
        """
        self.cfg = transform
        self.use_v2 = use_v2
        self.is_train = is_train

        if self.cfg["name"] == "sew_gesture":
            self.transform = self.sew_augment
        elif self.cfg["name"] == "Flip_Rotation":
            self.transform = self.default_argument

    @property
    def default_argument(self):
        """Generate the default transformation pipeline based on the configuration.
        
        Returns:
            T.Compose: Composed transformations.
        """
        T = get_module(self.use_v2)
        transforms = []
        if self.is_train:
            # transforms.append(T.RandomRotation(degrees=15))  # type: ignore
            transforms.append(T.RandomHorizontalFlip())
            # transforms.append(T.Resize(size=(48, 48)))
            # transforms.append(T.RandomVerticalFlip())
        else:
            # transforms.append(T.Resize(size=(48, 48)))
            pass
        return T.Compose(transforms)

    @property
    def sew_augment(
        self,
    ):
        if self.is_train:

            def argument(
                img,
            ):
                sec_list = np.random.choice(
                    img.shape[0], self.cfg["T_train"], replace=False
                )
                sec_list.sort()
                img = img[:, sec_list]
                return img

            return argument

        else:
            T = get_module(self.use_v2)
            return T.Compose([])

    def __call__(self, img: numpy.ndarray, *args: Any, **kwds: Any) -> Any:
        """Applies the defined data augmentation pipeline to the input image.

        Args:
            img (numpy.ndarray):  Input image to be augmented.

        Returns:
            Any: Augmented image after applying the defined transformations.
        """
        return self.transform(torch.from_numpy(img))
