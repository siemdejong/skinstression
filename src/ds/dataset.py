from pathlib import Path
from typing import Any, Optional
import os
import logging

from scipy.ndimage import convolve, convolve1d
from utils import get_lds_kernel_window, sturge

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms


log = logging.getLogger(__name__)


class THGStrainStressDataset(Dataset[Any]):
    def __init__(
        self,
        split: str,
        root_data_dir: str,
        folder: int,
        targets: np.ndarray,
        extension: str = "bmp",
        target_transform=None,
        reweight="none",
        lds=False,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
    ):
        # header = 0, assume there is a header in the labels.csv file.
        self.split = split
        self._data = Path(root_data_dir) / str(folder)
        self.group = folder
        self.targets = targets
        self.extension = extension
        self.transform = self.get_transform()
        self.target_transform = target_transform
        self._length = sum(1 for _ in os.listdir(self._data))

        self.weights = self._prepare_weights(
            reweight=reweight,
            lds=lds,
            lds_kernel=lds_kernel,
            lds_ks=lds_ks,
            lds_sigma=lds_sigma,
            param_ids=["a", "k", "xc"],
        )

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
            targets = self.target_transform(self.targets)

        weight = (
            np.asarray([self.weights[idx]]).astype("float32")
            if self.weights is not None
            else np.asarray([np.float32(1.0)])
        )

        return image, targets, weight

    def _prepare_weights(
        self,
        reweight: str,
        lds: bool = False,
        lds_kernel: str = "gaussian",
        lds_ks: int = 5,
        lds_sigma: int = 2,
        param_ids: Optional[str] = None,
    ):
        """Adaptation of label density smoothing (LDS) as described in arXiv:2102.09554v2.
        The adaptation allows for multi-dimensional label smoothening,
        under the assumption that the LDS kernel is separable like a 3D gaussian filter.

        Args:
            reweight: Choose none, inverse or square root inverse weighting ({none, inverse, sqrt_inv})
            lds: Switch to turn LDS on or off. Default: False.
            lds_kernel: The kernel to be used with density estimation see `get_lds_kernel_window()`. Default: gaussian.
            lds_ks: Kernel size. Default: 5.
            lds_sigma: Kernel distribution scale. See `get_lds_kernel_window()`. Default: 2.
            param_ids: parameters to choose from the THGStrainStressDataset.df. If None, all columns are selected. Default: None.
        """
        assert reweight in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        # Calculate count statistic.
        hist = []
        for column in param_ids:
            param_data = self.df[column].to_numpy()
            bins = sturge(len(param_data))
            column_hist, _ = np.histogram(param_data, bins=bins)
            hist.append(column_hist)

        if reweight == "sqrt_inv":
            hist = np.sqrt(hist)
        elif reweight == "inverse":
            # clip weights for inverse re-weight
            hist = np.clip(hist, -np.inf, np.inf)
            raise NotImplementedError

        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f"Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})")
            for i, label_count in enumerate(hist):
                hist[i] = convolve1d(
                    label_count,
                    weights=lds_kernel_window,
                    mode="constant",
                )

        weights = 1 / hist
        scaling = weights.size[1] / np.sum(weights, axis=0)
        scaled_weights = scaling * weights
        transposed_scaled_weights = scaled_weights.T

        return transposed_scaled_weights

    def get_transform(self):
        if self.split == "train":
            transform = transforms.Compose(
                [
                    # TODO: Insert some data augmentation transforms.
                    transforms.ToTensor(),
                    transforms.Grayscale()
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale()
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        return transform

    @staticmethod
    def load_data(
        split: str, data_path: str, targets_path: str
    ) -> tuple[Dataset, np.ndarray]:
        """Gather all data and construct a full dataset object.
        Return person_ids with it for stratification purposes.

        Args:
            data_path: the path to the directory containing the data.
            targets_path: the path to the file containing the target data.

        Returns:
            (tuple): tuple containing a concatenated dataset including all data indexed by targets_path,
            and an array of indices denoting to what group subsets of the dataset belong to.
        """
        assert split in {"train", "validation", "test"}
        datasets = []
        person_ids = []
        for _, labels in pd.read_csv(targets_path).iterrows():

            folder = int(labels["index"])
            targets = labels[["a", "k", "xc"]].to_numpy(dtype=float)
            person_id = labels[["person_id"]].to_numpy(dtype=float)

            if not (Path(data_path) / str(folder)).is_dir():
                log.info(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because it is not found."
                )
                continue

            dataset = THGStrainStressDataset(
                split=split,
                root_data_dir=data_path,
                folder=folder,
                targets=targets,
            )

            # Dirty way of checking if data is compatible with model.
            valid_shape = (1, 1000, 1000)
            if not dataset[0][0].shape == valid_shape:
                log.info(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because the data of size {dataset[0][0].shape} "
                    "is incompatible with the model."
                    f"Images of shape {valid_shape} are accepted."
                )
                continue

            datasets.append(dataset)
            person_ids.append(person_id)

        person_ids = np.array(person_ids)
        dataset = ConcatDataset(datasets)

        return dataset, person_ids
