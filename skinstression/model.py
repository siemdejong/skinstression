import warnings
from functools import partial
from typing import Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import Regressor, resnet10
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
# from skinstression.dataset import inverse_standardize

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*overflow encountered in exp*"
)


def logistic(x, a, k, xc):
    a, k, xc = a.to(torch.float), k.to(torch.float), xc.to(torch.float)
    return a / (1 + np.exp(-k * (x - xc)))


def plot_curve_pred(preds, curves):
    preds = preds.cpu()
    # standardization = {"mean": 3.8103177043244214, "std": 3.3450981056172124}
    # preds = inverse_standardize(preds, **standardization)
    for pred, strain, stress in zip(preds, curves["strain"], curves["stress"]):
        x = torch.linspace(1, 1.7, 70)
        (l,) = plt.plot(x, logistic(x, *pred))

        plt.scatter(strain.cpu(), stress.cpu(), color=l.get_color())

    plt.xlabel("Strain")
    plt.ylabel("Stress [MPa]")
    plt.xlim([1, 1.6])

    return plt.gcf()


class Skinstression(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0,
        out_size: int = 3,
    ) -> None:
        super().__init__()
        self.example_input_array = torch.randn((1, 1, 500, 500))
        backbone = resnet10(n_input_channels=1, spatial_dims=2)
        regressor = Regressor((1, 400), (out_size,), [1, 1, 1], [2, 2, 2])
        self.model = nn.Sequential(backbone, regressor)
        self.validation_step_outputs_preds = []
        self.validation_step_outputs_strain = []
        self.validation_step_outputs_stress = []
        self.lr = lr
        self.weight_decay = weight_decay

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
        loss, preds = self._common_step(batch)
        self.log("loss/val", loss, batch_size=batch["y"].shape[0])
        # strains, stresses = batch["curve"]["strain"], batch["curve"]["stress"]
        # curves = {"strains": strains, "stresses": stresses}
        # self.validation_step_outputs_preds.extend(preds)
        # self.validation_step_outputs_strain.extend(curves["strains"])
        # self.validation_step_outputs_stress.extend(curves["stresses"])

    def on_validation_epoch_end(self) -> None:
        preds = torch.stack(self.validation_step_outputs_preds)
        strain = self.validation_step_outputs_strain
        stress = self.validation_step_outputs_stress
        curves = {"strain": strain, "stress": stress}
        # self.logger.experiment.add_figure(
        #     "Val curves", plot_curve_pred(preds, curves), self.current_epoch
        # )
        if preds.shape[1] == 3:
            wandb.log({"chart": wandb.Image(plot_curve_pred(preds, curves))})
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_strain.clear()
        self.validation_step_outputs_stress.clear()

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("loss/test", loss, batch_size=batch[1].shape[0])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, 500, 1e-5)
        return [optimizer], [scheduler]
        # return optimizer

    def forward(self, x):
        return self.model(x)
