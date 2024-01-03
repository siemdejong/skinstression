"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""

from pathlib import Path

import lightning.pytorch as pl
from dataset import SkinstressionDataModule
from model import Skinstression

trainer = pl.Trainer(max_time="00:00:03:00", accelerator="gpu", devices=1)

model = Skinstression()
dm = SkinstressionDataModule(
    data_dir=Path("/home/sdejong/skinstression/data/stacks"),
    test_targets=Path("/home/sdejong/skinstression/data/splits/test.csv"),
    batch_size=2,
    num_workers=10,
)

trainer.test(
    ckpt_path="/home/sdejong/skinstression/lightning_logs/version_3204330/checkpoints/last.ckpt",
    model=model,
    datamodule=dm,
)
