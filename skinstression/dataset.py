from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from monai.data import CSVDataset, Dataset, DatasetFunc, ImageDataset, SmartCacheDataset
from monai.transforms import (
    CenterSpatialCrop,
    RandAdjustContrast,
    RandAxisFlip,
    RandRotate90,
    RandSpatialCrop,
    ResizeWithPadOrCrop,
)
import zarr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate, default_convert
from torchvision.transforms import Compose


# TODO: apply standardization to the targets.
def standardize(value: float, mean: float, std: float, eps: float = 1e-9) -> float:
    if std == 0:
        return value - mean
    return (value - mean) / (std + eps)




class SkinstressionDataset(Dataset):
    def __init__(self, images, params, usecols, curve_dir, sample_to_person):
        self.images = zarr.open(images, mode="r")
        self.image_keys = list(self.images.keys())
        self.curves = {curve.stem: pd.read_csv(curve) for curve in Path(curve_dir).glob("*.csv")}
        self.params = pd.read_csv(params, usecols=["sample_id"] + usecols)
        self.sample_to_person = pd.read_csv(sample_to_person)
        self.sample_ids = self.params["sample_id"]
        self.indices, self.cumsum = self.calc_indices_and_filter()
    
    def calc_indices_and_filter(self):
        num = 0
        lengths = []
        for img in self.images.keys():
            try:
                self.curves[str(img)]
            except KeyError:
                self.image_keys.remove(img)
                print(f"Removed {img} from dataset")
            else:
                length = self.images[img].shape[0]
                num += length
                lengths.append(length)
        cumsum = np.cumsum(lengths)
        return num, cumsum
    
    def __len__(self):
        return self.indices
    
    def __getitem__(self, index: int | slice | Sequence[int]):
        img_idx = np.digitize(index, self.cumsum)
        sample_id = self.image_keys[img_idx]
        slice_idx = img_idx - self.cumsum[np.digitize(index, self.cumsum)]
        img = self.images[sample_id][slice_idx, ...]
        target = self.params.loc[self.params["sample_id"] == int(sample_id)]
        curve = self.curves[str(sample_id)]
        return img, target, curve, sample_id

class SkinstressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images: Path | str,
        params: Path | str,
        curve_dir: Path | str,
        sample_to_person: Path | str,
        variables: list = ["A", "k", "xc"],
        batch_size: int = 1,
        num_workers: int = 0,
        standardization: dict = None,
        n_splits: int = 0,
        fold: int = 0,
        seed: int = 42,
        cache: bool = False,
        cache_num: int | None = None,
    ) -> None:
        super().__init__()

        self.images = Path(images)
        self.params = Path(params)
        self.curve_dir = Path(curve_dir)
        self.sample_to_person = Path(sample_to_person)
        self.variables = variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardization = standardization
        self.seed = seed
        self.fold = fold
        self.n_splits = n_splits
        self.cache = cache
        self.cache_num = cache_num

        if batch_size > 1:
            raise NotImplementedError(
                "Batch size larger than 1 is not supported yet."
            )

    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def make_splits(self, plot: bool = False) -> None:

        # Load and inspect the targets.
        df_params = pd.read_csv(self.params)
        df_persons = pd.read_csv(self.sample_to_person)
        df = df_params.merge(df_persons, on="sample_id")
        df.columns = map(str.lower, df.columns)

        df["filename"] = [str(index) + ".tif" for index in df["sample_id"]]
        filenames = list(
            str(fn.name) for fn in self.images.glob("*.tif")
        )  # Make sure the image is there!
        print(f"there are {len(df)}/{len(filenames)} eligible samples")

        # Split the full dataset in train and test.
        gss = GroupShuffleSplit(1, test_size=int(0.05 * len(df)), random_state=42)
        for split in gss.split(df["sample_id"], groups=df["person_id"]):
            super_train, test = split
            super_train_df = df[df["sample_id"].isin(super_train)]
            test_df = df[df["sample_id"].isin(test)]

        gss = GroupShuffleSplit(
            1, test_size=int(0.1 * len(super_train_df)), random_state=42
        )
        for split in gss.split(
            super_train_df["sample_id"], groups=super_train_df["person_id"]
        ):
            train, val = split
            train_df = df[df["sample_id"].isin(train)]
            val_df = df[df["sample_id"].isin(val)]

        print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

        if plot:
            raise NotImplementedError(
                "Plotting cross validation scheme is not supported yet."
            )

        return train_df, val_df, test_df

    def setup(self, stage: str):
        train_df, val_df, test_df = self.make_splits()
        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = SkinstressionDataset(
            params=self.params,
            usecols=self.variables,
            sample_to_person=self.sample_to_person,
            images=self.images,
            curve_dir=self.curve_dir,
        )

        if self.cache:
            self.train_dataset = SmartCacheDataset(
                self.train_dataset,
                len(self.train_dataset) if self.cache_num is None else self.cache_num,
            )
            self.val_dataset = SmartCacheDataset(
                self.val_dataset,
                len(self.val_dataset) if self.cache_num is None else self.cache_num,
            )
        
    def collate_fn(self, batch):
        batch_dict = {}
        batch_dict["img"] = torch.stack([torch.tensor(sample[0]) for sample in batch])
        batch_dict["target"] = torch.stack([torch.tensor(list(sample[1].values())) for sample in batch])
#         batch_dict["target"] = torch.stack([torch.tensor(list(sample[1].values())) for sample in batch])
#                                                           ^^^^^^^^^^^^^^^^^^
# TypeError: 'numpy.ndarray' object is not callable
        curves = [sample[2] for sample in batch]
        batch_dict["curve"] = [
            dict(zip(curve.T.index, curve.T.values)) for curve in curves
        ]
        batch_dict["sample_info"] = [sample[3] for sample in batch]
        return batch_dict

    # def collate_fn(self, batch):
    #     batch_dict = {}
    #     batch_dict["img"] = torch.stack(
    #         default_collate([sample["img"] for sample in batch])
    #     )
    #     batch_dict["target"] = torch.stack(
    #         [torch.tensor(list(sample["target"].values())) for sample in batch]
    #     )
    #     batch_dict["sample_info"] = [sample["sample_info"] for sample in batch]
    #     if "curve" in batch[0]:
    #         curves = [sample["curve"] for sample in batch]
    #         batch_dict["curve"] = [
    #             dict(zip(curve.T.index, curve.T.values)) for curve in curves
    #         ]
    #     return batch_dict

    # def collate_fn(self, batch):
    # batch = default_convert(batch)

    # lengths_central = torch.tensor([len(sample[0]) for sample in batch])
    # max_length_central = torch.max(lengths_central)
    # for sample in batch:
    #     central_tmp = torch.zeros(
    #         (max_length_central, *sample[0].shape[1:])
    #     )
    #     central_tmp[:len(sample[0])] = sample[0]
    #     sample[0] = central_tmp

    # batch_dict = {}
    # # TODO: select best slices and feed slices as a list to the 2D model instead of selecting only the first
    # batch_dict["img"] = default_collate([sample[0][:, :, 0].unsqueeze(0).unsqueeze(0) for sample in batch])
    # batch_dict["strain"] = [sample[2] for sample in batch]
    # batch_dict["stress"] = [sample[3] for sample in batch]
    # batch_dict["y"] = torch.tensor([sample[1] for sample in batch])

    # return batch_dict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
