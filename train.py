from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers.wandb import WandbLogger

from skinstression.dataset import SkinstressionDataModule
from skinstression.model import Skinstression
from skinstression.utils import cli_license_notice


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


if __name__ == "__main__":

    print(cli_license_notice)

    logger = WandbLogger(project="skinstression")

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=40,
        check_val_every_n_epoch=10,
        max_epochs=500,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[
            ModelCheckpoint(dirpath="checkpoints", monitor="loss/val"),
            LearningRateMonitor("epoch"),
        ],
    )

    model = Skinstression(backbone="resnet", model_depth=101)
    dm = SkinstressionDataModule(
        h5_path=Path("data/stacks.h5t"),
        batch_size=8,
        num_workers=0,
        n_splits=5,
        fold=0,  # Make sure to choose 0:n_splits-1 and don't change n_splits when doing cross-validation.
    )

    trainer.fit(model=model, datamodule=dm)
