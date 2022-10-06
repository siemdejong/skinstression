from pathlib import Path
from typing import Any
import os

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Grayscale,
    AugMix,
    RandAugment,
    FiveCrop,
    RandomCrop,
    AutoAugmentPolicy,
    AutoAugment,
)


# from load_data import load_image_data, load_label_data


class THGStrainStressDataset(Dataset[Any]):
    def __init__(
        self,
        root_data_dir: str,
        folder: int,
        targets: np.ndarray,
        extension: str = "bmp",
        data_transform=None,
        target_transform=None,
    ):
        # header = 0, assume there is a header in the labels.csv file.
        self._data = Path(root_data_dir) / str(folder)
        self.group = folder
        self.targets = targets
        self.extension = extension
        self.transform = data_transform
        self.target_transform = target_transform
        self._length = sum(1 for _ in os.listdir(self._data))

    def __len__(self):
        return self._length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Make it work with z-stacks!!!
        # https://stackoverflow.com/a/60176057
        # Assuming images follow [0, n-1], so they can be accesed directly.
        # data_path = self.data_dir / (str(int(self.labels["index"].iloc[idx])) + ".tif")
        image = Image.open(self._data / f"{str(idx)}.{self.extension}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            self.targets = self.target_transform(self.targets)

        return image, self.targets


def create_dataloader(
    batch_size: int,
    data_path: Path,
    label_path: Path,
    shuffle: bool = True,
) -> DataLoader[Any]:
    return DataLoader(
        dataset=THGStrainStressDataset(
            data_dir=data_path,
            label_path=label_path,
            data_transform=Compose(
                [
                    RandomCrop(size=(258, 258)),
                    # Resize((258, 258)),
                    Grayscale(),
                    # AugMix(),
                    # RandAugment(num_ops=2),
                    ToTensor(),
                    # Lambda(lambda y: (y - y.mean()) / y.std()), # To normalize the image.
                ],
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
