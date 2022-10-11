import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

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
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        self.epoch_count = 0
        self.loader = loader
        self.loss_metric = Metric()
        self.model = model
        self.compute_loss = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.stage = stage

    @property
    def avg_loss(self):
        return self.loss_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):

        # Turn on eval or train mode.
        self.model.train(self.stage is Stage.TRAIN)

        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            x, y = x.to(self.device), y.to(self.device)
            loss, batch_loss = self._run_single(x, y)

            experiment.add_batch_metric("loss", batch_loss, self.epoch_count)

            if self.optimizer:
                # Backpropagation
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    def _run_single(self, x: Any, y: Any):
        self.epoch_count += 1
        batch_size: int = len(x)
        prediction = self.model(x)
        loss = self.compute_loss(prediction.float(), y.float())

        self.prediction = prediction.detach().cpu().numpy()
        self.target = y.detach().cpu().numpy()

        batch_loss = loss.detach().cpu().numpy().mean()

        # Compute Batch Validation Metrics
        self.loss_metric.update(loss.item(), batch_size)
        return loss, batch_loss

    def reset(self):
        self.loss_metric = Metric()

    def save_checkpoint(self):
        torch.save(
            {
                "epoch": self.epoch_count,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{os.getcwd()}/checkpoint.pt",  # os.getcwd() is set by Hydra to 'outputs'.
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch_count = checkpoint["epoch"]


def run_test(
    test_runner: Runner,
    experiment: ExperimentTracker,
) -> None:
    # Testing Loop
    experiment.set_stage(Stage.TEST)
    test_runner.run("Test Batches", experiment)

    # Log Testing Epoch Metrics
    experiment.add_epoch_sigmoid(test_runner.prediction, test_runner.target)


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
    experiment.add_epoch_sigmoid(train_runner.prediction, train_runner.target, epoch_id)

    # Validation Loop
    experiment.set_stage(Stage.VAL)
    with torch.no_grad():
        val_runner.run("Validation Batches", experiment)

    # Log Validation Epoch Metrics
    experiment.add_epoch_sigmoid(val_runner.prediction, val_runner.target, epoch_id)

    # Combine training and validation loss in one plot.
    loss_value_dict = {"train": train_runner.avg_loss, "val": val_runner.avg_loss}
    experiment.add_epoch_metrics("loss", loss_value_dict, epoch_id)


def run_fold(
    val_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    scheduler: torch.optim.lr_scheduler,
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
    if len(runners) > 1:
        summary = f"[Epoch: {epoch_id + 1}/{epoch_count}]"
    else:
        summary = f"Testing results after {epoch_count} epochs"
    for runner in runners:
        if runner.stage == Stage.TRAIN:
            loss_msg = f"Train Loss: {runner.avg_loss: 0.4f}"
        elif runner.stage == Stage.VAL:
            loss_msg = f"Validation Loss: {runner.avg_loss: 0.4f}"
        elif runner.stage == Stage.TEST:
            loss_msg = f"Test Loss: {runner.avg_loss: 0.4f}"
        summary = ", ".join([summary, loss_msg])

    return summary
