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


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        stage: Stage,
        scaler,
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.local_rank = local_rank
        self.stage = stage
        self.disable_progress_bar = not progress_bar
        self.scaler = scaler
        self.dry_run = dry_run

    @property
    def avg_loss(self):
        return self.loss_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):

        # Turn on eval or train mode.
        self.model.train(self.stage is Stage.TRAIN)

        for batch_num, (x, y, w) in enumerate(
            tqdm(self.loader, desc=desc, ncols=80, disable=self.disable_progress_bar)
        ):
            # if self.dry_run and self.run_count:
            #     break
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
            dist.all_reduce(
                loss
            )  # TODO: MAKE SURE THIS ALL REDUCE LOSS IS WHAT WE WANT (AND NOT E.G. AVG)

            logging.debug(f"    iteration: {batch_num} | loss: {loss}")

            # Compute Batch Validation Metrics
            self.loss_metric.update(loss.detach(), len(x))

            if self.local_rank == 0:
                experiment.add_batch_metric("loss", loss.detach(), self.run_count)

        # Only let 1 process save checkpoints.
        # Check only every epoch.
        if self.local_rank == 0 and self.optimizer:
            if self.should_save(loss):
                self.save_checkpoint()

        if self.scheduler and self.optimizer:
            self.scheduler.step()

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

    def should_save(self, loss):
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            return True
        else:
            return False

    def save_checkpoint(self):
        state: dict[str, Union[int, dict[str, Any]]] = {
            "epoch": self.run_count,
            "model_state_dict": self.model.module.state_dict(),
        }

        if self.optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scaler.is_enabled():
            state["scaler_state_dict"] = self.scaler.state_dict()

        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(
            state,
            f"{os.getcwd()}/checkpoint.pt",  # os.getcwd() is set by Hydra to 'outputs'.
        )
        logging.info("Checkpoint saved.")

    def load_checkpoint(self, path: str):
        # NOTE: consume_prefix_in_state_dict_if_present() should be used
        # if loading from a DDP saved checkpoint.
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.run_count = checkpoint["epoch"]

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
    experiment: ExperimentTracker,
    epoch_id: int,
    local_rank: int,
) -> None:
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    if local_rank == 0:
        # Log Training Epoch Metrics
        experiment.add_epoch_logistic_curve(
            train_runner.prediction.detach().cpu(),
            train_runner.target.detach().cpu(),
            epoch_id,
        )

    # TODO: Possibly only do validation once every x epochs.
    # Validation Loop
    experiment.set_stage(Stage.VAL)
    with torch.no_grad():
        val_runner.run("Validation Batches", experiment)

    if local_rank == 0:
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

        logging.info(
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
