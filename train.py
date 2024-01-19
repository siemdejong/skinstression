"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from skinstression.dataset import SkinstressionDataModule
from skinstression.model import Skinstression
from skinstression.utils import cli_license_notice

# TODO: make user independent defaults.
config_defaults = dict(
    # Config
    images="data/stacks.zarr",
    curve_dir="data/curves/",
    params="data/params.csv",
    sample_to_person="data/sample_to_person.csv",
    max_epochs=100,
    log_every_n_steps=100,
    check_val_every_n_epoch=1,
    precision="bf16-mixed",
    n_splits=5,
    fold=0,  # Make sure to choose 0:n_splits-1 and don't change n_splits when doing cross-validation.
    variables=["a", "k", "xc"],
    # variables=["k"],  # Alternative strategy: train three models, one for each variable.
    cache=True,
    num_workers=8,
    # Search space
    batch_size_exp=0,
    lr=1e-4,
    weight_decay=1e-2,
    proj_hidden_dim_exp=11,
    local_proj_hidden_dim_exp=7,
)

wandb.init(config=config_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def train_function(config):

    logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=config["log_every_n_steps"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        callbacks=[
            LearningRateMonitor("epoch"),
            ModelCheckpoint(monitor="loss/val", mode="min"),
        ],
    )

    model = Skinstression(
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        out_size=len(config["variables"]),
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
        cache=config["cache"],
        num_workers=config["num_workers"],
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    print(cli_license_notice)
    print(f"Starting a run with {config}")
    train_function(config)
