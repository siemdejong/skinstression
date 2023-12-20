import os
from pathlib import Path
from typing import Union, Literal

from functools import partial

import torch
from torch.utils.data.dataloader import default_convert, default_collate
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import h5torch
from monai.transforms import RandAxisFlip, RandRotate90, RandAdjustContrast, ResizeWithPadOrCrop, CenterSpatialCrop, RandSpatialCrop
from torchvision.transforms import Compose
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import pandas as pd
from torch.utils.data import DataLoader

# TODO: apply standardization to the targets.
def standardize(value: float, mean: float, std: float, eps: float = 1e-9) -> float:
    if std == 0:
        return value - mean
    return (value - mean) / (std + eps)

def inverse_standardize(value: float, mean: float, std: float) -> float:
    return value * std + mean

class SkinstressionDataset(h5torch.Dataset):
    def __init__(self, set: Literal["train", "val", "test"], *args, **kwargs) -> None:
        self.transform = self.get_transform(set)
        super().__init__(*args, **kwargs)

    def get_transform(self, set):
        transforms = [
            ResizeWithPadOrCrop((31, 1000, 1000))]
        if set == "train":
            transforms.extend([
                CenterSpatialCrop((31, 700, 700)),
                RandSpatialCrop((10, 500, 500), random_size=False),
                RandAxisFlip(prob=0.25),
                RandRotate90(prob=1/4, spatial_axes=(1, 2)),
                RandAdjustContrast(prob=1, gamma=(1, 0.4)),
            ])
        else:
            transforms.extend([
                CenterSpatialCrop((10, 500, 500)),
            ])
        transforms = Compose(transforms)
        return transforms
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["central"] = self.transform(torch.tensor(sample["central"]).unsqueeze(0))
        return sample

class SkinstressionDataModule(pl.LightningDataModule):

    def __init__(self, h5_path: Path, path_curves: Path, train_targets: Union[Path, None] = None, val_targets: Union[Path, None] = None, test_targets: Union[Path, None] = None, batch_size: int = 1, num_workers: int = 0, standardization: dict = None, n_splits: int = 0, fold: int = 0, seed: int = 42) -> None:
        super().__init__()

        self.h5_path = h5_path
        self.path_curves = path_curves
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.test_targets = test_targets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardization = standardization
        self.seed = seed
        self.fold = fold
        self.n_splits = n_splits

        if num_workers > 0:
            raise NotImplementedError("num_workers > 0 is not supported yet")
    
    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def make_splits(self, plot: bool = False) -> None:        
        full_dataset = h5torch.Dataset(str(self.h5_path))

        gss = GroupShuffleSplit(1, random_state=self.seed)
        gkf = GroupKFold(self.n_splits)

        # Make the super split into train and test.
        train_super_idx, test_idx = next(gss.split(full_dataset.indices, groups=[sample["0/person_id"] for sample in full_dataset]))
        full_dataset.close()

        # Make the train split into train and val.
        dataset = h5torch.Dataset(str(self.h5_path), subset=train_super_idx)
        train_idx, val_idx = list(gkf.split(dataset.indices, groups=[sample["0/person_id"] for sample in dataset]))[self.fold]
        dataset.close()

        if plot:
            raise NotImplementedError("Plotting cross validation scheme is not supported yet.")

        return train_idx, val_idx, test_idx

    def setup(self, stage: str):
        train_idx, val_idx, test_idx = self.make_splits()
        if stage == "fit":
            self.train_dataset = SkinstressionDataset(set="train", file=str(self.h5_path), subset=train_idx)
            self.val_dataset = SkinstressionDataset(set="val", file=str(self.h5_path), subset=val_idx)
        elif stage == "test":
            self.test_dataset = SkinstressionDataset(set="test", file=str(self.h5_path), subset=test_idx)

    def collate_fn(self, batch):
        batch = default_convert(batch)
        
        lengths_central = torch.tensor([len(sample["central"]) for sample in batch])
        max_length_central = torch.max(lengths_central)
        for sample in batch:
            central_tmp = torch.zeros((max_length_central, *sample["central"].shape[1:]))
            central_tmp[:len(sample["central"])] = sample["central"]
            sample["central"] = central_tmp
        
        batch_dict = {}
        batch_dict["central"] = default_collate([sample["central"] for sample in batch])
        batch_dict["strain"] = [sample["0/strain"] for sample in batch]
        batch_dict["stress"] = [sample["0/stress"] for sample in batch]
        a = [sample["0/a"] for sample in batch]
        k = [sample["0/k"] for sample in batch]
        xc = [sample["0/xc"] for sample in batch]
        batch_dict["y"] = torch.tensor([a, k, xc]).T
        
        return batch_dict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, pin_memory=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, pin_memory=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, pin_memory=True, num_workers=self.num_workers, collate_fn=self.collate_fn)
