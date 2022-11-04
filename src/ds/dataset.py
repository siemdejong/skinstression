import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils.data import ConcatDataset, Dataset

from ds.utils import get_lds_kernel_window, sturge


class THGStrainStressDataset(Dataset[Any]):
    def __init__(
        self,
        split: str,
        root_data_dir: str,
        folder: int,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        extension: str = "bmp",
        target_transform=None,
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

        self.weights = weights

    def __len__(self):
        return self._length

    def __getitem__(self, idx) -> tuple[torch.tensor, np.ndarray, np.ndarray]:
        # TODO: Make it work with z-stacks!!!
        # https://stackoverflow.com/a/60176057
        # Assuming images follow [0, n-1], so they can be accesed directly.
        # data_path = self.data_dir / (str(int(self.labels["index"].iloc[idx])) + ".tif")
        image = Image.open(self._data / f"{str(idx)}.{self.extension}")
        targets = self.targets

        weight = self.weights  # (
        #     np.asarray([self.weights]).astype("float32")
        #     if self.weights is not None
        #     else np.asarray([np.float32(1.0)])
        # )

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            targets = self.target_transform(targets)

        return image, targets, weight

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
            param_ids: parameters to choose from the THGStrainStressDataset.df. If None, all columns are selected. Default: None.
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

            print(param, len(edges), len(emp_target_dist))

            if reweight == "sqrt_inv":
                emp_target_dist = np.sqrt(emp_target_dist)
            elif reweight == "inv":
                # Clip weights for inverse re-weight
                emp_target_dist = np.clip(emp_target_dist, -np.inf, np.inf)
                raise NotImplementedError

            # Calculate effective label distribution
            if param_ks:
                lds_ks = param_ks.get(param)
            else:
                lds_ks = 5
            if param_sigma:
                lds_sigma = param_sigma.get(param)
            else:
                lds_sigma = 2
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

        return (
            np.asarray(all_weights),
            np.asarray(all_emp_dists, dtype=object),
            np.asarray(all_eff_dists, dtype=object),
            np.asarray(all_edges, dtype=object),
        )

        # Calculate count statistic.
        bin_edges_list = []
        bin_count_list = []
        for param_data in targets.T:
            # bins = sturge(len(param_data))
            # bins = 500
            bin_edges = np.histogram_bin_edges(param_data, bins=500)
            bin_edges_list.append(bin_edges)

            idcs = np.digitize(param_data, bin_edges)
            bin_count = np.bincount(idcs)
            bin_count_list.append(bin_count)
        bin_count_list = np.asarray(bin_count_list)

        if reweight == "sqrt_inv":
            bin_count_list = np.sqrt(bin_count_list)
        elif reweight == "inverse":
            # clip weights for inverse re-weight
            bin_count_list = np.clip(bin_count_list, -np.inf, np.inf)
            raise NotImplementedError

        idcs_list = []
        num_per_target = []
        num_per_bin_list = []
        for i, target in enumerate(targets.T):
            idcs = np.digitize(target, bin_edges_list[i])
            idcs_list.append(idcs)

            num_per_bin = np.bincount(idcs)
            num_per_bin_list.append(num_per_bin)

            target_as_bin_count = num_per_bin[idcs]
            num_per_target.append(target_as_bin_count)

        logging.info(f"Using re-weighting: [{reweight.upper()}]")
        num_per_target_original = num_per_target
        smoothed_num_per_bin_list = []
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            logging.info(
                f"Using LDS: {lds_kernel.upper()} | kernel size={lds_ks} | sigma={lds_sigma})"
            )
            num_per_target = []
            for i, target in enumerate(targets.T):
                num_per_bin = num_per_bin_list[i]
                smoothed_num_per_bin = convolve1d(
                    num_per_bin,
                    weights=lds_kernel_window,
                    mode="constant",
                )

                smoothed_num_per_bin_list.append(smoothed_num_per_bin)

                # Bin edges haven't changed. No need to redigitize target.
                idcs = idcs_list[i]
                target_as_smoothed_bin_count = smoothed_num_per_bin[idcs]
                num_per_target.append(target_as_smoothed_bin_count)

        num_per_target = np.asarray(num_per_target).T
        weights = 1 / num_per_target
        scaling = weights.shape[1] / np.sum(weights, axis=0)
        scaled_weights = scaling * weights

        return scaled_weights, num_per_target_original, num_per_target

    def get_transform(self):
        if self.split == "train":
            transform = transforms.Compose(
                [
                    # TODO: Insert some data augmentation transforms.
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        return transform

    @staticmethod
    def load_data(
        split: str,
        data_path: str,
        targets_path: str,
        reweight="none",
        lds=False,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
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

        weights, _, _ = THGStrainStressDataset._prepare_weights(
            targets=targets_list,
            reweight=reweight,
            lds=lds,
            lds_kernel=lds_kernel,
            lds_ks=lds_ks,
            lds_sigma=lds_sigma,
            param_ids=["a", "k", "xc"],
        )

        # Build dataset.
        datasets = []
        person_ids = []
        for (_, labels), weights in zip(pd.read_csv(targets_path).iterrows(), weights):

            folder = int(labels["index"])
            person_id = labels[["person_id"]].to_numpy(dtype=float)
            targets = labels[["a", "k", "xc"]].to_numpy(dtype=float)

            if not (Path(data_path) / str(folder)).is_dir():
                logging.error(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because it is not found."
                )
                continue

            dataset = THGStrainStressDataset(
                split=split,
                root_data_dir=data_path,
                folder=folder,
                targets=targets,
                weights=weights,
            )

            # Dirty way of checking if data is compatible with model.
            valid_shape = (1, 1000, 1000)
            if not dataset[0][0].shape == valid_shape:
                logging.error(
                    f"{Path(data_path) / str(folder)} will be excluded "
                    f"because the data of size {dataset[0][0].shape} "
                    "is incompatible with the model."
                    f"Images of shape {valid_shape} are accepted."
                )
                continue

            datasets.append(dataset)
            person_ids.extend([person_id] * len(dataset))

        person_ids = np.array(person_ids)
        dataset = ConcatDataset(datasets)

        return dataset, person_ids
