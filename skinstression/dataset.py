from pathlib import Path
from typing import Union, Literal

from functools import partial

import ast
import cv2
import torch
import numpy as np
from monai.transforms import RandAxisFlip, RandRotate90, RandAdjustContrast, ResizeWithPadOrCrop, CenterSpatialCrop, RandSpatialCrop
from torchvision.transforms import Compose
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def standardize(value: float, mean: float, std: float, eps: float = 1e-9) -> float:
    if std == 0:
        return value - mean
    return (value - mean) / (std + eps)

def inverse_standardize(value: float, mean: float, std: float) -> float:
    return value * std + mean

class SkinstressionDataset(Dataset):
    def __init__(self, image_dir: Path, path: Path, path_curves: Path, set: Literal["train", "val", "test", "inference"], standardization: dict[Literal["mean", "std"], float] = {"mean": 0, "std": 0}) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.df = pd.read_csv(path)
        self.df_curves = pd.read_csv(path_curves, index_col="index", converters={1:ast.literal_eval, 2:ast.literal_eval})

        transforms = [ResizeWithPadOrCrop((31, 1000, 1000))]
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
        self.transforms = Compose(transforms)

        self.target_transform = partial(standardize, **standardization)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        curve_tmp = self.df_curves.loc[sample["index"]]
        curve = {"strain": torch.zeros((857,)), "stress": torch.zeros((857,))}
        curve["strain"][:len(curve_tmp["strain"])] = torch.tensor(curve_tmp["strain"])
        curve["stress"][:len(curve_tmp["stress"])] = torch.tensor(curve_tmp["stress"])

        target = torch.tensor([float(sample["a"]), float(sample["k"]), float(sample["xc"])])
        target = self.target_transform(target)

        path = str(self.image_dir / sample["filename"])
        image = torch.tensor(np.array(cv2.imreadmulti(path)[1]), dtype=torch.float)
        output = self.transforms(image.unsqueeze(0))

        return output, target, curve

    def __len__(self):
        return len(self.df)

class SkinstressionDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: Path, path_curves: Path, train_targets: Union[Path, None] = None, val_targets: Union[Path, None] = None, test_targets: Union[Path, None] = None, batch_size: int = 1, num_workers: int = 0, standardization: dict=None) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.path_curves = path_curves
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.test_targets = test_targets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardization = standardization
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SkinstressionDataset(self.data_dir, self.train_targets, self.path_curves, set="train", standardization=self.standardization)
            self.val_dataset = SkinstressionDataset(self.data_dir, self.val_targets, self.path_curves, set="val", standardization=self.standardization)
        elif stage == "test":
            self.test_dataset = SkinstressionDataset(self.data_dir, self.test_targets, self.path_curves, set="test", standardization=self.standardization)
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, pin_memory=True)
