from pathlib import Path
from typing import Any
import os
import logging

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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


log = logging.getLogger(__name__)


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

    def __getitem__(self, idx) -> tuple[Image.Image, np.ndarray]:
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

    @staticmethod
    def load_data(data_path: str, targets_path: str) -> tuple[Dataset, np.ndarray]:
        """Gather all data and construct a full dataset object. Return groups with it.

        Args:
            data_path: the path to the directory containing the data.
            targets_path: the path to the file containing the target data.

        Returns:
            (tuple): tuple containing a concatenated dataset including all data indexed by targets_path,
            and an array of indices denoting to what group subsets of the dataset belong to.
        """
        data_transform = Compose(
            [
                Grayscale(),
                ToTensor(),
            ]
        )
        datasets = []
        groups = []
        for _, labels in pd.read_csv(targets_path).iterrows():

            folder = int(labels["index"])
            targets = labels[["A", "h", "slope", "C"]].to_numpy(dtype=float)

            if not (Path(data_path) / str(folder)).is_dir():
                log.info(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because it is not found."
                )
                continue

            dataset = THGStrainStressDataset(
                root_data_dir=data_path,
                folder=folder,
                targets=targets,
                data_transform=data_transform,
            )

            # Dirty way of checking if data is compatible with model.
            if not dataset[0][0].shape == (1, 1000, 1000):
                log.info(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because the data of size {dataset[0][0].shape} "
                    "is incompatible with the model."
                )
                continue

            datasets.append(dataset)
            groups.extend([folder] * len(dataset))

        groups = np.array(groups)
        dataset = ConcatDataset(datasets)

        return dataset, groups
