from array import array
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard.writer import SummaryWriter

from ds.tracking import Stage
from ds.utils import create_experiment_log_dir
from ds.functions import sigmoid
from typing import Optional


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
        self, prediction: np.ndarray, target: np.ndarray, step: Optional[int] = None
    ) -> matplotlib.figure.Figure:
        x = np.linspace(1, 2.5, 1000)

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
        ax.set_ylim(0, 7)
        ax.legend()

        return fig
