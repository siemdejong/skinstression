"""Implements Pytorch Dataset.
Copyright (C) 2022  Siem de Jong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import convolve1d
from scipy import stats
from torch.utils.data import ConcatDataset, Dataset

from ds.utils import get_lds_kernel_window
from ds.transforms import YeoJohnsonTransform
from ds.exceptions import IOErrorAfterRetries

log = logging.getLogger(__name__)


class SkinstressionDataset(Dataset[Any]):
    def __init__(
        self,
        split: str,
        root_data_dir: str,
        folder: int,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        extension: str = "bmp",
        target_transform=None,
        top_k: Optional[int] = None,
        importances: Optional[np.ndarray[float]] = None,
    ):
        self.split = split
        self._data = Path(root_data_dir) / str(folder)
        self.group = folder
        self.targets = targets
        self.extension = extension
        self.transform = self.get_transform()
        self.target_transform = target_transform
        self.top_k = top_k
        self._length = self.calc_length()
        self.weights = weights
        self.importances = importances

    def calc_length(self):
        # Exclude files from PyIQ.
        length_whole_dataset = len(glob(f"{self._data}/*.{self.extension}"))

        if self.top_k:
            # To make sure top_k is not too large.
            if self.top_k > length_whole_dataset:
                log.error(f"Config top_k too large. Using full dataset {self._data}.")
            number = min(self.top_k, length_whole_dataset)
        else:
            number = length_whole_dataset

        return number

    def __len__(self):
        return self._length

    def __getitem__(self, idx) -> tuple[torch.tensor, np.ndarray, np.ndarray]:
        # TODO: Make it work with z-stacks!!!
        # https://stackoverflow.com/a/60176057
        # Assuming images follow [0, n-1], so they can be accesed directly.
        # data_path = self.data_dir / (str(int(self.labels["index"].iloc[idx])) + ".tif")
        data_path = self._data / f"{str(idx)}.{self.extension}"
        _max_attempts = 10
        for attempt in range(_max_attempts):
            try:
                image = Image.open(data_path)
            except OSError:
                log.error(
                    f"Opening image throws OSError. Retrying... Attempt {attempt}"
                )
                import time

                time.sleep(10)  # Retry later.
                continue
            else:
                break
        else:  # If the image really cannot be opened after several times.
            raise IOErrorAfterRetries(_max_attempts, data_path)

        targets = self.targets
        weights = self.weights
        importances = self.importances

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            targets = self.target_transform(targets)

        return image, targets, weights, importances

    @staticmethod
    def _prepare_weights(
        targets: np.ndarray,
        reweight: str,
        lds: bool = False,
        lds_kernel: str = "gaussian",
        param_roi: Optional[dict[str, int]] = None,
        param_ks: Optional[dict[str, int]] = None,
        param_sigma: Optional[dict[str, int]] = None,
        param_desc: Optional[str] = None,
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
        """
        assert reweight in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        all_weights = []
        all_emp_dists = []
        all_eff_dists = []
        all_edges = []
        for param_data, param in zip(targets.T, param_desc):
            if param_roi:
                edges = param_roi.get(param)
            else:
                edges = np.histogram_bin_edges(param_data, "auto")
            bin_index_per_target = np.digitize(param_data, edges)

            # minlength so extrapolation is possible.
            emp_target_dist = np.bincount(bin_index_per_target, minlength=len(edges))

            log.info(f"Using re-weighting: [{reweight.upper()}]")
            if reweight == "sqrt_inv":
                emp_target_dist = np.sqrt(emp_target_dist)
            elif reweight == "inv":
                # Clip weights for inverse re-weight
                emp_target_dist = np.clip(emp_target_dist, -np.inf, np.inf)
                raise NotImplementedError

            if lds:
                # Calculate effective label distribution
                if param_ks:
                    lds_ks = param_ks.get(param)
                else:
                    lds_ks = 5
                if param_sigma:
                    lds_sigma = param_sigma.get(param)
                else:
                    lds_sigma = 2
                log.info(
                    f"LDS: {param} "
                    + f"| kernel: {lds_kernel.upper()} "
                    + f"| param roi: {edges[0]:.2f}-{edges[-1]:.2f} "
                    + f"| param res.: {edges[1]-edges[0]:.5f} "
                    + f"| kernel size: {lds_ks} "
                    + f"| sigma: {lds_sigma}"
                )
                lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
                eff_target_dist = convolve1d(
                    emp_target_dist, weights=lds_kernel_window, mode="constant"
                )

                eff_num_per_target = np.float32(eff_target_dist[bin_index_per_target])
                weights = 1 / eff_num_per_target

                scaling = len(weights) / np.sum(weights)
                scaled_weights = scaling * weights

                all_weights.append(scaled_weights)
                all_emp_dists.append(emp_target_dist)
                all_eff_dists.append(eff_target_dist)
                all_edges.append(edges)
            else:
                weights = 1 / np.float32(emp_target_dist[bin_index_per_target])
                scaling = len(weights) / np.sum(weights)
                scaled_weights = scaling * weights
                all_weights.append(scaled_weights)
                all_emp_dists.append(emp_target_dist)
                all_eff_dists.append(None)
                all_edges.append(edges)

        return (
            np.asarray(all_weights).T,
            np.asarray(all_emp_dists, dtype=object),
            np.asarray(all_eff_dists, dtype=object),
            np.asarray(all_edges, dtype=object),
        )

    def get_transform(self):

        if self.split == "train":
            transform = transforms.Compose(
                [
                    # NOTE: The Yeo-Johnson transform and normalization are time-consuming.
                    # It is possible to do them in preprocessing e.g. in notebook 17.
                    # YeoJohnsonTransform(0.466319593487972),
                    transforms.ToTensor(),
                    # NOTE: If using values below, be careful to not leak information
                    # from the val/test sets to the training set.
                    # NOTE: These mean and std are statistics after the Yeo-Johnson transform.
                    # transforms.Normalize(
                    #     mean=(14.716653741103862 / 255),
                    #     std=(6.557034596034911 / 255),
                    # ),
                    # TODO: Make sure changing the crop aspect ratio doesn't change physics.
                    transforms.Resize(
                        (1000, 1000)
                    ),  # Stupid hack to accept images smaller than 1000x1000.
                    transforms.RandomCrop((700, 700)),
                    # transforms.RandomResizedCrop(
                    #     size=(258, 258), scale=(0.9, 1), ratio=(1, 1)
                    # ),
                    transforms.Resize((258, 258)),
                    transforms.ColorJitter(brightness=0.3),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # YeoJohnsonTransform(0.466319593487972),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=(14.716653741103862 / 255),
                    #     std=(6.557034596034911 / 255),
                    # ),
                    transforms.Resize((1000, 1000)),
                    transforms.CenterCrop((700, 700)),
                    transforms.Resize((258, 258)),
                ]
            )
        return transform

    @staticmethod
    def load_data(
        split: str,
        data_path: str,
        targets_path: str,
        top_k: int,
        reweight="none",
        importances: Optional[np.ndarray[float]] = None,
        lds=False,
        lds_kernel="gaussian",
        extension="bmp",
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

        # Pre-calculate weights.
        targets_list = []
        for _, labels in pd.read_csv(targets_path).iterrows():
            targets_list.append(labels[["a", "k", "xc"]].to_numpy())
        targets_list = np.asarray(targets_list)

        param_roi = {
            "a": np.arange(0, 15, 0.01),
            "k": np.arange(0, 50, 0.1),
            "xc": np.arange(1, 1.5, 0.001),
        }
        param_ks = {
            "a": 30,
            "k": 30,
            "xc": 30,
        }
        param_sigma = {
            "a": 3,
            "k": 3,
            "xc": 3,
        }

        weights, _, _, _ = SkinstressionDataset._prepare_weights(
            targets=targets_list,
            reweight=reweight,
            lds=lds,
            lds_kernel=lds_kernel,
            param_roi=param_roi,
            param_ks=param_ks,
            param_sigma=param_sigma,
            param_desc=["a", "k", "xc"],
        )

        # Build dataset.
        datasets = []
        person_ids = []
        for (_, labels), weights in zip(pd.read_csv(targets_path).iterrows(), weights):

            folder = int(labels["index"])
            person_id = labels[["person_id"]].to_numpy(dtype=float)
            targets = labels[["a", "k", "xc"]].to_numpy(dtype=float)

            if not (Path(data_path) / str(folder)).is_dir():
                log.error(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because it is not found."
                )
                continue

            dataset = SkinstressionDataset(
                split=split,
                root_data_dir=data_path,
                folder=folder,
                targets=targets,
                weights=weights,
                top_k=top_k,
                importances=importances,
                extension=extension,
            )

            # Dirty way of checking if data is compatible with model.
            # valid_shape = (1, 1000, 1000)
            # if not dataset[0][0].shape == valid_shape:
            #     log.error(
            #         f"{Path(data_path) / str(folder)} will be excluded "
            #         f"because the data of size {dataset[0][0].shape} "
            #         "is incompatible with the model."
            #         f"Images of shape {valid_shape} are accepted."
            #     )
            #     continue

            datasets.append(dataset)
            person_ids.extend([person_id] * len(dataset))

        person_ids = np.array(person_ids)
        dataset = ConcatDataset(datasets)

        return dataset, person_ids
