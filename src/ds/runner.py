"""Provides metric class for Runner to keep metric history.
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

import logging
import os
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ds.metrics import Metric
from ds.tracking import ExperimentTracker, Stage
from ds.utils import reduce_tensor

log = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        stage: Stage,
        scaler: torch.cuda.amp.GradScaler,
        global_rank: int,
        local_rank: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        progress_bar: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.loss_metric = Metric()
        self.model = model
        self.compute_loss = loss_fn
        self.lowest_loss = np.inf
        self.lowest_loss_restart = np.inf
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.next_lr = 0  # TODO: this is not actually 0...
        self.restart_count = 0
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.stage = stage
        self.disable_progress_bar = not progress_bar
        self.scaler = scaler
        self.dry_run = dry_run

    @property
    def avg_loss(self):
        return self.loss_metric.average

    def run(self, desc: str, experiment: ExperimentTracker, epoch_id):

        # Turn on eval or train mode.
        self.model.train(self.stage is Stage.TRAIN)

        self.epoch_id = epoch_id

        for batch_num, (x, y, w) in enumerate(
            tqdm(self.loader, desc=desc, ncols=80, disable=self.disable_progress_bar)
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.scaler.is_enabled(),
            ):

                x, y, w = (
                    x.to(self.local_rank),
                    y.to(self.local_rank),
                    w.to(self.local_rank),
                )
                loss = self._run_single(x, y, w)

            if self.optimizer:
                # Backpropagation
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.model.zero_grad(set_to_none=True)

            # The distributed validation losses must be reduced to the same loss.
            # Doesn't seem to be implemented for the scaler.scale().backward() call, sadly.
            reduce_tensor(loss)

            # Compute Batch Validation Metrics
            self.loss_metric.update(loss.detach(), len(x))

            if self.global_rank == 0:
                log.debug(f"    iteration: {batch_num} | loss: {loss}")
                experiment.add_batch_metric("loss", loss.detach(), self.run_count)

            if self.dry_run:
                break

        if self.scheduler and self.optimizer:
            self.current_lr = self.next_lr
            self.scheduler.step()
            self.next_lr = self.scheduler.get_last_lr()[0]

        # Only let 1 process save checkpoints.
        # Checkpoint only once every epoch at the maximum.
        if self.global_rank == 0:
            self.save_checkpoint()

    def _run_single(self, x: Any, y: Any, w: float):
        """
        Args:
            w: weighting of the loss function.
        """
        self.run_count += 1
        self.prediction = self.model(x)
        self.target = y
        loss = self.compute_loss(self.prediction, self.target, w)

        return loss

    def reset(self):
        self.loss_metric = Metric()

    def should_save(self) -> str:
        """Determine if a checkpoint should be saved because of a warm restart or a lowest loss."""

        filenames = []

        # Only train runners hold scheduler information.
        if self.stage is Stage.TRAIN:
            restart = self.next_lr > self.current_lr
            if restart:
                self.restart_count += 1
                self.lowest_loss_restart = np.inf

        # Only save lowest loss checkpoint based on validation loss.
        elif self.stage is Stage.VAL:
            new_low = self.loss_metric.average < self.lowest_loss
            if new_low:
                self.lowest_loss = self.loss_metric.average
                filenames.append("low")

            new_low_restart = self.loss_metric.average < self.lowest_loss_restart
            if new_low_restart:
                self.lowest_loss_restart = self.loss_metric.average
                filenames.append(f"low_restart-{self.restart_count}")

        return filenames

    def save_checkpoint(self) -> None:

        # First check if the checkpoint needs saving.
        # Return checkpoint filename.
        filenames = self.should_save()

        # should_save returns empty list if saving is not needed.
        for checkpoint_fn in filenames:
            state: dict[str, Union[int, dict[str, Any]]] = {
                "epoch": self.epoch_id,
                "model_state_dict": self.model.state_dict(),
            }

            if self.optimizer:
                state["optimizer_state_dict"] = self.optimizer.state_dict()

            if self.scaler.is_enabled():
                state["scaler_state_dict"] = self.scaler.state_dict()

            if self.scheduler:
                state["scheduler_state_dict"] = self.scheduler.state_dict()

            torch.save(
                state,
                f"{os.getcwd()}/{checkpoint_fn}.pt",  # os.getcwd() is set by Hydra to 'outputs'.
            )
            log.info(f"Checkpoint '{checkpoint_fn}' saved at epoch {self.epoch_id}.")

    def load_checkpoint(self, path: str):
        # NOTE: consume_prefix_in_state_dict_if_present() should be used
        # if loading a DDP saved checkpoint to non-DDP.
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch_id = checkpoint["epoch"]

        # Might be needed in the future (https://github.com/pytorch/pytorch/issues/2830):
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda(gpus)


def run_test(
    test_runner: Runner,
    experiment: ExperimentTracker,
) -> None:
    # Testing Loop
    experiment.set_stage(Stage.TEST)
    test_runner.run("Test Batches", experiment)

    # Log Testing Epoch Metrics
    experiment.add_epoch_logistic_curve(
        test_runner.prediction.detach(), test_runner.target.detach()
    )


def run_epoch(
    val_runner: Runner,
    train_runner: Runner,
    epoch_id: int,
    experiment: Optional[ExperimentTracker] = None,
) -> None:

    # Training Loop
    if experiment:  # only global rank 0 does experiment tracking.
        experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment, epoch_id)

    if experiment:
        # Log Training Epoch Metrics
        experiment.add_epoch_param("lr", train_runner.current_lr, epoch_id)
        experiment.add_epoch_logistic_curve(
            train_runner.prediction.detach().cpu(),
            train_runner.target.detach().cpu(),
            epoch_id,
        )

    # TODO: Possibly only do validation once every x epochs.
    # Validation Loop
    if experiment:
        experiment.set_stage(Stage.VAL)
    with torch.no_grad():
        val_runner.run("Validation Batches", experiment, epoch_id)

    if experiment:
        # Log Validation Epoch Metrics
        experiment.add_epoch_logistic_curve(
            val_runner.prediction.detach().cpu(),
            val_runner.target.detach().cpu(),
            epoch_id,
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

            # TODO: Only save checkpoints at rank 0.
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
