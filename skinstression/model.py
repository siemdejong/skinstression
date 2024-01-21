"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from monai.networks.nets import Regressor, resnet10
from torch.optim import SGD

__all__ = ["Skinstression"]


class Skinstression(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0,
        momentum: float = 0,
        out_size: int = 3,
    ) -> None:
        super().__init__()
        self.example_input_array = torch.randn((1, 1, 500, 500))
        backbone = resnet10(n_input_channels=1, spatial_dims=2)
        regressor = Regressor((1, 400), (out_size,), [1, 1, 1], [2, 2, 2])
        self.model = nn.Sequential(backbone, regressor)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def _common_step(self, batch):
        x, y = batch["img"], batch["target"]
        pred = self.forward(x)
        loss = nn.functional.l1_loss(pred, y)
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch)
        self.log("loss/train", loss, batch_size=batch["target"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch)
        self.log("loss/val", loss, batch_size=batch["target"].shape[0])

    def test_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch)
        self.log("loss/test", loss, batch_size=batch["target"].shape[0])

    def predict_step(self, batch, batch_idx):
        _, pred = self._common_step(batch)
        return pred

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        return optimizer

    def forward(self, x):
        return self.model(x).flatten(start_dim=1)
