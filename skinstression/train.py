from pathlib import Path
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, LearningRateFinder
import torch

from model import Skinstression
from dataset import SkinstressionDataModule

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

trainer = pl.Trainer(max_epochs=500, accelerator="gpu", devices=1, precision="bf16-mixed", callbacks=[ModelCheckpoint(dirpath="/home/sdejong/skinstression/checkpoints_v3", monitor="val_loss"), LearningRateMonitor("epoch"), StochasticWeightAveraging(swa_lrs=1e-2)])#, FineTuneLearningRateFinder(torch.arange(100, 1000, 50))])

model = Skinstression()
model = torch.compile(model)
dm = SkinstressionDataModule(
    data_dir=Path(sys.argv[1]),
    train_targets=Path("/home/sdejong/skinstression/data/splits/fold-1-split-train.csv"),
    val_targets=Path("/home/sdejong/skinstression/data/splits/fold-1-split-val.csv"),
    path_curves=Path("/home/sdejong/skinstression/data/curves.csv"),
    batch_size=32,
    num_workers=14,
    standardization={"mean": 3.8103177043244214, "std": 3.3450981056172124},
)

trainer.fit(ckpt_path="/home/sdejong/skinstression/checkpoints_v3/epoch=98-step=99.ckpt", model=model, datamodule=dm)
