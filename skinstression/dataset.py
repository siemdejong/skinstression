from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import h5torch
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
from monai.transforms.io.array import SUPPORTED_READERS
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate, default_convert
from torchvision.transforms import Compose


# TODO: apply standardization to the targets.
def standardize(value: float, mean: float, std: float, eps: float = 1e-9) -> float:
    if std == 0:
        return value - mean
    return (value - mean) / (std + eps)


def inverse_standardize(value: float, mean: float, std: float) -> float:
    return value * std + mean


class SkinstressionDatasetv1(h5torch.Dataset):
    def __init__(self, set: Literal["train", "val", "test"], *args, **kwargs) -> None:
        self.transform = self.get_transform(set)
        super().__init__(*args, **kwargs)

    def get_transform(self, set):
        transforms = [ResizeWithPadOrCrop((31, 1000, 1000))]
        if set == "train":
            transforms.extend(
                [
                    CenterSpatialCrop((31, 700, 700)),
                    RandSpatialCrop((10, 500, 500), random_size=False),
                    RandAxisFlip(prob=0.25),
                    RandRotate90(prob=1 / 4, spatial_axes=(1, 2)),
                ]
            )
        else:
            transforms.extend(
                [
                    CenterSpatialCrop((10, 500, 500)),
                ]
            )
        transforms = Compose(transforms)
        return transforms

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["central"] = self.transform(torch.tensor(sample["central"]).unsqueeze(0))
        return sample


class SkinstressionDatasetv2(ImageDataset):
    def __init__(self, image_dir, curves_dir, df, variables, *args, **kwargs) -> None:
        self.df = df.reset_index()
        image_files = (Path(image_dir) / df["filename"]).tolist()
        self.curves_dir = Path(curves_dir)
        assert isinstance(
            variables, list
        ), "no variables selected. E.g. variables=['a', 'k', 'xc']"
        labels = torch.tensor(df[list(variables)].to_numpy())
        super().__init__(image_files=image_files, labels=labels, *args, **kwargs)

    def __getitem__(self, index):
        sample = list(super().__getitem__(index))
        sample[0] = sample[0].unsqueeze(
            0
        )  # Because the model expects a batch and color dimension, next to the 3D image.
        sample_id = str(self.df.loc[index]["sample_id"])
        df_curves = pd.read_csv(
            str(self.curves_dir / Path(sample_id).with_suffix(".csv"))
        )
        sample.append(df_curves["strain"].to_numpy())
        sample.append(df_curves["stress"].to_numpy())
        return tuple(sample)


class SkinstressionDataset(Dataset):
    def __init__(
        self,
        params,
        cols,
        sample_to_person,
        image_dir,
        curve_dir=None,
        suffix=".tif",
        reader="ITKReader",
    ):
        """
        cols: list of params (str) to be selected from csv files in params. Order must correspond with model output.
        """
        self.target_dataset = CSVDataset(
            [str(params), str(sample_to_person)],
            col_names=["sample_id", "person_id"] + sorted(cols),
        )
        if curve_dir is not None:
            curve_files = list(Path(curve_dir).glob("*.csv"))
            self.curves_datasets = []
            for curve_file in curve_files:
                curve_dataset = pd.read_csv(curve_file)
                self.curves_datasets.append(curve_dataset)
        else:
            self.curves_datasets = None
        image_files = list(Path(image_dir).glob(f"*{suffix}"))
        image_files = DatasetFunc(
            image_files,
            self.filter_and_sort_ineligible_data,
            csv_dataset=self.target_dataset,
        )
        self.img_dataset = ImageDataset(image_files, reader=reader)
        self.sample_info = self.pop_sample_infos()
        self.data = image_files

    def pop_sample_infos(self):
        sample_infos = []
        for data in self.target_dataset:
            sample_info = {
                "sample_id": data.pop("sample_id"),
                "person_id": data.pop("person_id"),
            }
            sample_infos.append(sample_info)
        return sample_infos

    @staticmethod
    def filter_and_sort_ineligible_data(data: list[Path], csv_dataset: CSVDataset):
        eligible_sample_ids = map(lambda x: x["sample_id"], csv_dataset)
        eligible_data = []
        for _id in eligible_sample_ids:
            for sample in data:
                if int(sample.stem) != _id:
                    continue
                eligible_data.append(str(sample))
        return eligible_data

    def __getitem__(self, index: int | slice | Sequence[int]):
        out = {
            "img": self.img_dataset[index],
            "target": self.target_dataset[index],
            "sample_info": self.sample_info[index],
        }
        if self.curves_datasets is not None:
            out["curve"] = self.curves_datasets[index]
        return out


class SkinstressionDataModulev1(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Path,
        variables: list = ["a", "k", "xc"],
        batch_size_exp: int = 3,
        num_workers: int = 0,
        standardization: dict = None,
        n_splits: int = 0,
        fold: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.h5_path = Path(h5_path)
        self.variables = variables
        self.batch_size = 2**batch_size_exp
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
        train_super_idx, test_idx = next(
            gss.split(
                full_dataset.indices,
                groups=[sample["0/person_id"] for sample in full_dataset],
            )
        )
        full_dataset.close()

        # Make the train split into train and val.
        dataset = h5torch.Dataset(str(self.h5_path), subset=train_super_idx)
        train_idx, val_idx = list(
            gkf.split(
                dataset.indices, groups=[sample["0/person_id"] for sample in dataset]
            )
        )[self.fold]
        dataset.close()

        if plot:
            raise NotImplementedError(
                "Plotting cross validation scheme is not supported yet."
            )

        return train_idx, val_idx, test_idx

    def setup(self, stage: str):
        train_idx, val_idx, test_idx = self.make_splits()
        if stage == "fit":
            self.train_dataset = SkinstressionDataset(
                set="train", file=str(self.h5_path), subset=train_idx
            )
            self.val_dataset = SkinstressionDataset(
                set="val", file=str(self.h5_path), subset=val_idx
            )
        elif stage == "test":
            self.test_dataset = SkinstressionDataset(
                set="test", file=str(self.h5_path), subset=test_idx
            )

    def collate_fn(self, batch):
        batch = default_convert(batch)

        lengths_central = torch.tensor([len(sample["central"]) for sample in batch])
        max_length_central = torch.max(lengths_central)
        for sample in batch:
            central_tmp = torch.zeros(
                (max_length_central, *sample["central"].shape[1:])
            )
            central_tmp[: len(sample["central"])] = sample["central"]
            sample["central"] = central_tmp

        batch_dict = {}
        batch_dict["central"] = default_collate([sample["central"] for sample in batch])
        batch_dict["strain"] = [sample["0/strain"] for sample in batch]
        batch_dict["stress"] = [sample["0/stress"] for sample in batch]
        targets = [[sample[f"0/{var}"] for var in self.variables] for sample in batch]
        # a = [sample["0/a"] for sample in batch]
        # k = [sample["0/k"] for sample in batch]
        # xc = [sample["0/xc"] for sample in batch]
        batch_dict["y"] = torch.tensor(targets)

        return batch_dict

    def collate_fn2(self, batch):
        batch["curve"] = dict(zip(batch["curve"].T.index, batch["curve"].T.values))
        print(batch)
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn2,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn2,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn2,
        )


class SkinstressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: Path | str,
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
        reader: SUPPORTED_READERS = "ITKReader",
    ) -> None:
        super().__init__()

        self.image_dir = Path(image_dir)
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
        self.reader = reader
        self.cache = cache
        self.cache_num = cache_num

    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def make_splits(self, plot: bool = False) -> None:

        # Load and inspect the targets.
        df_params = pd.read_csv(Path(self.image_dir) / "params.csv")
        df_persons = pd.read_csv(Path(self.image_dir) / "sample_to_person.csv")
        df = df_params.merge(df_persons, on="sample_id")
        df.columns = map(str.lower, df.columns)

        df["filename"] = [str(index) + ".tif" for index in df["sample_id"]]
        filenames = list(
            str(fn.name) for fn in self.image_dir.glob("*.tif")
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
        # SkinStressionDataset = partial(
        #     SkinstressionDataset(
        #         image_dir=self.image_dir,
        #         cols=self.variables,
        #         curves_dir=self.curves_dir,

        #     )
        # )
        # if stage == "fit":
        #     self.train_dataset = SkinstressionDataset(image_dir=self.image_dir, curves_dir=self.curves_dir, df=train_df, variables=self.variables)
        #     self.val_dataset = SkinstressionDataset(image_dir=self.image_dir, curves_dir=self.curves_dir, df=val_df, variables=self.variables)
        # elif stage == "test":
        #     self.test_dataset = SkinstressionDataset(image_dir=self.image_dir, curves_dir=self.curves_dir, df=test_df, variables=self.variables)
        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = SkinstressionDataset(
            params=self.params,
            cols=self.variables,
            sample_to_person=self.sample_to_person,
            image_dir=self.image_dir,
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
        batch_dict["img"] = torch.stack(
            default_collate([sample["img"] for sample in batch])
        )
        batch_dict["target"] = torch.stack(
            [torch.tensor(list(sample["target"].values())) for sample in batch]
        )
        batch_dict["sample_info"] = [sample["sample_info"] for sample in batch]
        if "curve" in batch[0]:
            curves = [sample["curve"] for sample in batch]
            batch_dict["curve"] = [
                dict(zip(curve.T.index, curve.T.values)) for curve in curves
            ]
        return batch_dict

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
