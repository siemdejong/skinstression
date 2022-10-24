import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from ds.metrics import Metric
from ds.tracking import ExperimentTracker, Stage
import os

log = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        stage: Stage,
        scaler,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = torch.device("cpu"),
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        progress_bar: Optional[bool] = False,
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.loss_metric = Metric()
        self.model = model
        self.compute_loss = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.stage = stage
        self.disable_progress_bar = not progress_bar
        self.scaler = scaler

    @property
    def avg_loss(self):
        return self.loss_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):

        # Turn on eval or train mode.
        self.model.train(self.stage is Stage.TRAIN)

        for x, y in tqdm(
            self.loader, desc=desc, ncols=80, disable=self.disable_progress_bar
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.scaler.is_enabled(),
            ):

                x, y = x.to(self.device), y.to(self.device)
                loss = self._run_single(x, y)

            experiment.add_batch_metric("loss", loss.detach(), self.run_count)

            if self.optimizer:
                # Backpropagation
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.model.zero_grad(set_to_none=True)

        if self.scheduler:
            self.scheduler.step()

    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size: int = len(x)
        self.prediction = self.model(x)
        self.target = y
        loss = self.compute_loss(self.prediction, y)

        # Compute Batch Validation Metrics
        self.loss_metric.update(loss.detach(), batch_size)

        return loss

    def reset(self):
        self.loss_metric = Metric()

    def save_checkpoint(self):
        state: dict[str, Union[int, dict[str, Any]]] = {
            "epoch": self.run_count,
            "model_state_dict": self.model.state_dict(),
        }

        if self.optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scaler.is_enabled():
            state["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(
            state,
            f"{os.getcwd()}/checkpoint.pt",  # os.getcwd() is set by Hydra to 'outputs'.
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.run_count = checkpoint["epoch"]


def run_test(
    test_runner: Runner,
    experiment: ExperimentTracker,
) -> None:
    # Testing Loop
    experiment.set_stage(Stage.TEST)
    test_runner.run("Test Batches", experiment)

    # Log Testing Epoch Metrics
    experiment.add_epoch_sigmoid(
        test_runner.prediction.detach(), test_runner.target.detach()
    )


def run_epoch(
    val_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch_id: int,
) -> None:
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    # Log Training Epoch Metrics
    experiment.add_epoch_sigmoid(
        train_runner.prediction.detach().cpu(),
        train_runner.target.detach().cpu(),
        epoch_id,
    )

    # Validation Loop
    experiment.set_stage(Stage.VAL)
    with torch.no_grad():
        val_runner.run("Validation Batches", experiment)

    # Log Validation Epoch Metrics
    experiment.add_epoch_sigmoid(
        val_runner.prediction.detach().cpu(), val_runner.target.detach().cpu(), epoch_id
    )

    # Combine training and validation loss in one plot.
    loss_value_dict = {"train": train_runner.avg_loss, "val": val_runner.avg_loss}
    experiment.add_epoch_metrics("loss", loss_value_dict, epoch_id)


def run_fold(
    val_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    fold_id: int,
    epoch_count: int,
) -> None:

    _lowest_loss = np.inf

    # Run the epochs
    for epoch_id in range(epoch_count):
        run_epoch(val_runner, train_runner, experiment, epoch_id)

        if val_runner.avg_loss < _lowest_loss:
            _lowest_loss = val_runner.avg_loss
            train_runner.save_checkpoint()

        experiment.add_fold_metric("loss", val_runner.avg_loss, fold_id)

        log.info(
            summary(
                train_runner,
                val_runner,
                epoch_id=epoch_id,
                epoch_count=epoch_count,
            )
        )

        # Reset the runners
        train_runner.reset()
        val_runner.reset()

        scheduler.step()

        # Flush the tracker after every epoch for live updates
        experiment.flush()

    # run_test(test_runner=test_runner, experiment=tracker)
    # print_summary(test_runner, epoch_count=EPOCH_COUNT)
    # test_runner.reset()
    # tracker.flush()


def summary(
    *runners, epoch_id: Optional[int] = None, epoch_count: Optional[int] = None
) -> str:
    if len(runners) > 1 and epoch_id:
        summary = f"[Epoch: {epoch_id + 1}/{epoch_count}]"
    else:
        summary = f"Testing results after {epoch_count} epochs"

    for runner in runners:
        loss_msg = ""
        if runner.stage == Stage.TRAIN:
            loss_msg = f"Train Loss: {runner.avg_loss: 0.4f}"
        elif runner.stage == Stage.VAL:
            loss_msg = f"Validation Loss: {runner.avg_loss: 0.4f}"
        elif runner.stage == Stage.TEST:
            loss_msg = f"Test Loss: {runner.avg_loss: 0.4f}"
        summary = ", ".join([summary, loss_msg])

    return summary
