"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""

from io import StringIO
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import zarr
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from monai.data import Dataset, SmartCacheDataset
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader


# TODO: apply standardization to the targets.
def standardize(value: float, mean: float, std: float, eps: float = 1e-9) -> float:
    if std == 0:
        return value - mean
    return (value - mean) / (std + eps)


class SkinstressionDataset(Dataset):
    def __init__(self, images, params, usecols, curve_dir, sample_to_person):
        self.images = zarr.open(images, mode="r")
        self.image_keys = list(self.images.keys())
        self.curves = {
            curve.stem: pd.read_csv(curve)
            for curve in sorted(Path(curve_dir).glob("*.csv"))
        }
        self.usecols = usecols
        self.params = pd.read_csv(params, usecols=["sample_id"] + usecols)
        self.sample_to_person = pd.read_csv(sample_to_person)
        self.sample_ids = self.params["sample_id"]
        self.indices, self.cumsum = self.calc_indices_and_filter()

    @classmethod
    def from_df(cls, df: pd.DataFrame, *args, **kwargs):
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        dataset = cls(params=buffer, *args, **kwargs)
        return dataset

    def calc_indices_and_filter(self):
        num = 0
        lengths = []
        for img in self.images.keys():
            try:
                self.curves[str(img)]
                if int(img) not in self.params["sample_id"].values.astype(int):
                    raise KeyError
            except KeyError:
                try:
                    self.image_keys.remove(img)
                except ValueError:
                    continue
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
        slice_idx = index - self.cumsum[np.digitize(index, self.cumsum)]
        img = self.images[sample_id][slice_idx, ...]
        target = self.params.loc[self.params["sample_id"] == int(sample_id)][
            self.usecols
        ]
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
                "Batch size larger than 1 is not supported yet. The model does not accept this yet."
            )

    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def make_splits(self, plot: bool = False) -> None:

        try:
            Path("tmp").mkdir()

            # Load and inspect the targets.
            df_params = pd.read_csv(self.params)
            df_persons = pd.read_csv(self.sample_to_person)
            df = df_params.merge(df_persons, on="sample_id")
            df.columns = list(map(str.lower, df.columns))

            images = zarr.open(self.images, mode="r")
            print(f"there are {len(df)}/{len(images)} eligible samples")

            # Split the full dataset in train and test.
            gss = GroupShuffleSplit(1, test_size=int(0.05 * len(df)), random_state=42)
            for split in gss.split(df["sample_id"], groups=df["person_id"]):
                super_train, test = split
                super_train_df = df[df["sample_id"].isin(super_train)]
                test_df = df[df["sample_id"].isin(test)]

            gss = GroupShuffleSplit(  # TODO: change to GroupKFold when cross validation is deemed important.
                1, test_size=int(0.1 * len(super_train_df)), random_state=42
            )
            for split in gss.split(
                super_train_df["sample_id"], groups=super_train_df["person_id"]
            ):
                train, val = split
                train_df = super_train_df[super_train_df["sample_id"].isin(train)]
                val_df = super_train_df[super_train_df["sample_id"].isin(val)]

            print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
            train_df.to_csv("tmp/train.csv", index=False)
            val_df.to_csv("tmp/val.csv", index=False)
            test_df.to_csv("tmp/test.csv", index=False)
        except FileExistsError:
            print("loading splits from tmp folder")
            train_df = pd.read_csv("tmp/train.csv")
            val_df = pd.read_csv("tmp/val.csv")
            test_df = pd.read_csv("tmp/test.csv")

        if plot:
            raise NotImplementedError(
                "Plotting cross validation scheme is not supported yet."
            )

        return train_df, val_df, test_df

    def setup(self, stage: str):
        train_df, val_df, test_df = self.make_splits()

        def _prep_dataset(df):
            dataset = SkinstressionDataset.from_df(
                df,
                images=self.images,
                usecols=self.variables,
                curve_dir=self.curve_dir,
                sample_to_person=self.sample_to_person,
            )
            return dataset

        if stage == "fit":
            self.train_dataset = _prep_dataset(train_df)
            self.val_dataset = _prep_dataset(val_df)
        elif stage == "test":
            self.test_dataset = _prep_dataset(test_df)

        if self.cache and stage == "fit":
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
        batch_dict["img"] = torch.stack(
            [
                torch.tensor(sample[0], dtype=torch.float).unsqueeze(0)
                for sample in batch
            ]
        )
        batch_dict["target"] = torch.stack(
            [torch.tensor(sample[1].to_numpy()) for sample in batch]
        ).flatten(start_dim=1)
        curves = [sample[2] for sample in batch]
        batch_dict["curve"] = [
            dict(zip(curve.T.index, curve.T.values)) for curve in curves
        ]
        batch_dict["sample_info"] = [sample[3] for sample in batch]
        return batch_dict

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
