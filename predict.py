"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from skinstression.dataset import SkinstressionDataModule
from skinstression.model import Skinstression
from skinstression.utils import cli_license_notice, plot_prediction

torch.multiprocessing.set_sharing_strategy("file_system")


def bin_per_sample(predictions_new, prediction, data):
    try:
        predictions_new[data["sample_info"][0]]["predictions"].append(prediction)
        predictions_new[data["sample_info"][0]]["data"].append(data)
    except KeyError:
        predictions_new[data["sample_info"][0]] = {
            "predictions": [prediction],
            "data": [data],
        }
    return predictions_new


# TODO: make user independent defaults.
config = dict(
    # Config
    images="data/stacks.zarr",
    curve_dir="data/curves/",
    params="data/params.csv",
    sample_to_person="data/sample_to_person.csv",
    ckpt_path="/mnt/c/Users/Z405155/Downloads/model.ckpt",
    n_splits=5,
    fold=0,  # Make sure to choose 0:n_splits-1 and don't change n_splits when doing cross-validation.
    variables=["a", "k", "xc"],
    # variables=["k"],  # Alternative strategy: train three models, one for each variable.
    num_workers=8,
    save_plots=True,
    # Search space
    batch_size_exp=0,
    proj_hidden_dim_exp=11,
    local_proj_hidden_dim_exp=7,
)


def train_function(config):
    trainer = pl.Trainer()
    model = Skinstression.load_from_checkpoint(
        checkpoint_path=config["ckpt_path"], out_size=len(config["variables"])
    )
    dm = SkinstressionDataModule(
        images=config["images"],
        curve_dir=config["curve_dir"],
        params=config["params"],
        sample_to_person=config["sample_to_person"],
        variables=config["variables"],
        batch_size=2 ** config["batch_size_exp"],
        n_splits=config["n_splits"],
        fold=config["fold"],
        num_workers=config["num_workers"],
    )

    predictions = trainer.predict(model=model, datamodule=dm)
    predictions_binned_per_sample = dict()
    for prediction, data in tqdm(
        zip(predictions, dm.predict_dataloader()),
        desc="organizing output",
        total=len(predictions),
    ):
        predictions_binned_per_sample = bin_per_sample(
            predictions_binned_per_sample, prediction, data
        )

    for sample_id, items in tqdm(
        predictions_binned_per_sample.items(),
        desc="plotting curves",
        total=len(predictions_binned_per_sample),
    ):
        predictions = items["predictions"]
        data_arr = items["data"]
        for prediction, data in tqdm(
            zip(predictions, data_arr),
            desc="plotting sample_id",
            total=len(predictions),
            leave=False,
        ):
            plot_prediction(
                pred=prediction.detach().cpu().numpy().T,
                target=data["target"].detach().cpu().numpy().T,
                slice_idx=data["slice_idx"][0],
                num_slices=len(predictions),
            )
        plt.title(sample_id)
        plt.tight_layout()
        if config["save_plots"]:
            plt.savefig(f"tmp/sample={sample_id}.png")
        else:
            plt.show()
        plt.clf()


if __name__ == "__main__":
    print(cli_license_notice)
    print(f"Starting prediction with {config}")
    train_function(config)
