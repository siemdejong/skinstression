from pathlib import Path
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from model import Skinstression
from dataset import SkinstressionDataModule

trainer = pl.Trainer(max_time="00:04:45:00", accelerator="gpu", devices=1, precision="bf16-mixed", callbacks=[ModelCheckpoint(dirpath="/home/sdejong/skinstression/checkpoints_v2", monitor="val_loss")])

model = Skinstression()
# model = torch.compile(model)
dm = SkinstressionDataModule(
    data_dir=Path(sys.argv[1]),
    # train_targets=Path("/home/sdejong/skinstression/data/splits/fold-1-split-train.csv"),
    # val_targets=Path("/home/sdejong/skinstression/data/splits/fold-1-split-val.csv"),
    path_curves=Path("/home/sdejong/skinstression/data/curves.csv"),
    train_targets=Path("/home/sdejong/skinstression/data/splits/overfit.csv"),
    val_targets=Path("/home/sdejong/skinstression/data/splits/overfit.csv"),
    batch_size=16,
    num_workers=14,
)

trainer.fit(model=model, datamodule=dm)
