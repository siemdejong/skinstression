"""Provides tensorboard experiment class to upload logs to tensorboard.
Copyright (C) 2022  Siem de Jong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

This file incorporates work covered by the following copyright and permission notice:  

    Copyright (c) 2021 Mark Todisco & ArjanCodes

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

from ds.functions import logistic, sigmoid
from ds.tracking import Stage
from ds.utils import create_experiment_log_dir


class TensorboardExperiment:
    def __init__(self, log_path: str, create: bool = True) -> None:

        log_dir = create_experiment_log_dir(root=log_path)
        self.stage = Stage.TRAIN
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

        charts = {
            f"training": [
                "Multiline",
                [f"TRAIN/epoch/loss", f"VAL/epoch/loss"],
            ]
        }
        layout = {"THG-STRAIN-STRESS": charts}
        self._writer.add_custom_scalars(layout)

    def set_stage(self, stage: Stage) -> None:
        self.stage = stage

    def flush(self) -> None:
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True) -> None:
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metrics(
        self, name: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        main_tag = f"epoch/{name}"
        self._writer.add_scalars(
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=step
        )
    
    def add_epoch_param(self, name: str, value: float, step: int) -> None:
        tag = f"epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_train_val_epoch_metrics(
        self, name: str, train_value: float, val_value: float, step: int
    ) -> None:
        main_tag = f"{Stage.TRAIN.name}-{Stage.VAL.name}/epoch/{name}"
        scalars = {"train": train_value, "val": val_value}
        self._writer.add_scalars(main_tag, scalars, step)

    def add_fold_metric(self, name: str, value: float, step: int) -> None:
        tag = f"{self.stage.name}/fold/{name}"
        self._writer.add_scalar(tag, value, step)

    # def add_epoch_confusion_matrix(
    #     self, y_true: list[np.array], y_pred: list[np.array], step: int
    # ):
    #     y_true, y_pred = self.collapse_batches(y_true, y_pred)
    #     fig = self.create_confusion_matrix(y_true, y_pred, step)
    #     tag = f"{self.stage.name}/epoch/confusion_matrix"
    #     self._writer.add_figure(tag, fig, step)

    # @staticmethod
    # def collapse_batches(
    #     y_true: list[np.array], y_pred: list[np.array]
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     return np.concatenate(y_true), np.concatenate(y_pred)

    # def create_confusion_matrix(
    #     self, y_true: list[np.array], y_pred: list[np.array], step: int
    # ) -> plt.Figure:
    #     cm = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues")
    #     cm.ax_.set_title(f"{self.stage.name} Epoch: {step}")
    #     return cm.figure_

    def add_epoch_logistic_curve(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        step=None,
    ) -> None:
        fig = self.create_logistic_curve(
            prediction=prediction, target=target, step=step
        )
        tag = f"{self.stage.name}/epoch/logistic_curve"
        self._writer.add_figure(tag, fig, step)

    def create_logistic_curve(
        self, prediction: torch.tensor, target: torch.tensor, step: Optional[int] = None
    ) -> matplotlib.figure.Figure:
        x = torch.linspace(1, 2.5, 1000)

        fig, ax = plt.subplots(1, 1)

        if step:
            ax.set_title(f"{self.stage.name} Epoch: {step}")
        else:
            ax.set_title(f"{self.stage.name}")

        # TODO: Fix colors.
        for idx, (y_prediction, y_target) in enumerate(zip(prediction, target)):
            ax.plot(x, logistic(x, *y_prediction), "--", label="prediction")
            ax.plot(x, logistic(x, *y_target), "-", label="target")

        ax.set_xlim(0.8, 2.6)
        ax.set_ylim(0, 14)
        ax.legend()

        return fig

    def add_epoch_sigmoid(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        step=None,
    ) -> None:
        fig = self.create_sigmoid(prediction=prediction, target=target, step=step)
        tag = f"{self.stage.name}/epoch/sigmoid"
        self._writer.add_figure(tag, fig, step)

    def create_sigmoid(
        self, prediction: torch.tensor, target: torch.tensor, step: Optional[int] = None
    ) -> matplotlib.figure.Figure:
        x = torch.linspace(1, 2.5, 1000)

        fig, ax = plt.subplots(1, 1)

        if step:
            ax.set_title(f"{self.stage.name} Epoch: {step}")
        else:
            ax.set_title(f"{self.stage.name}")

        # TODO: Fix colors.
        for idx, (y_prediction, y_target) in enumerate(zip(prediction, target)):
            ax.plot(x, sigmoid(x, *y_prediction), "--", label="prediction")
            ax.plot(x, sigmoid(x, *y_target), "-", label="target")

        ax.set_xlim(0.8, 2.6)
        ax.set_ylim(0, 14)
        ax.legend()

        return fig

    def add_hparams(self, hparams: dict[str, float], loss: Optional[float] = None):
        self._writer.add_hparams(hparams, {"hparam/loss": loss if loss else "Pruned"})
