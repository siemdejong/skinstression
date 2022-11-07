"""Provides trainer class to train a model.
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
"""

import logging
import os
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    LinearLR,
)
from torch.utils.data import random_split, Subset
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel as DDP

from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.models import THGStrainStressCNN
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment
from ds.loss import weighted_l1_loss
from ds.utils import seed_all, ddp_cleanup, ddp_setup
from ds.logging_setup import setup_worker_logging


class Trainer:
    def __init__(self, dataset, cfg: THGStrainStressConfig):
        self.cfg = cfg
        if self.cfg.try_overfit:
            self.train_subset = Subset(dataset, indices=[0, 1])
            self.val_subset = Subset(dataset, indices=[0, 1])
        else:
            # Split the dataset in train, validation and test (sub)sets.
            train_test_split = int(len(dataset) * 0.8)
            train_set, test_set = random_split(
                dataset, [train_test_split, len(dataset) - train_test_split]
            )

            train_val_split = int(len(train_set) * 0.8)
            self.train_subset, self.val_subset = random_split(
                train_set, [train_val_split, len(train_set) - train_val_split]
            )

    def __call__(self, local_rank: int, world_size: int):
        """
        Args:
            cfg: hydra configuration object.
        """
        model = THGStrainStressCNN(self.cfg)
        model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

        loss_fn = weighted_l1_loss  # MAE.
        optimizer = getattr(torch.optim, self.cfg.params.optimizer.name)(
            model.parameters(),
            lr=self.cfg.params.optimizer.lr,
            weight_decay=self.cfg.params.optimizer.weight_decay,
        )
        warmup_scheduler = LinearLR(
            optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=10
        )
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=400,
            T_mult=self.cfg.params.optimizer.T_mult,
        )
        scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        # Define dataloaders
        train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=int(self.cfg.params.batch_size),
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=int(self.cfg.params.batch_size),
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
            pin_memory=True,
        )

        # Create the runners
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
        )
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            scheduler=scheduler,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
        )

        # Setup the experiment tracker
        log_dir = os.getcwd() + "/tensorboard"
        tracker = TensorboardExperiment(log_path=log_dir)

        # Run epochs.
        for epoch_id in range(self.cfg.params.epoch_count):
            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                experiment=tracker,
                epoch_id=epoch_id,
                local_rank=0,
            )

            loss = val_runner.avg_loss
            logging.info(f"epoch: {epoch_id} | loss: {loss}")


def train(
    local_rank: int, world_size: int, cfg: THGStrainStressConfig, log_queue: Queue
):

    # Setup logging.
    setup_worker_logging(local_rank, log_queue, cfg.debug)

    # Set and seed device
    torch.cuda.set_device(local_rank)
    logging.info(f"Hello from device {local_rank} of {world_size}")
    seed_all(cfg.seed)

    # Initialize process group
    ddp_setup(local_rank, world_size)
    # Load dataset.
    dataset, groups = THGStrainStressDataset.load_data(
        split="train",
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
        reweight="sqrt_inv",
        lds=True,
    )
    trainer = Trainer(dataset, cfg)
    trainer(local_rank, world_size)

    # Exit subprocesses.
    ddp_cleanup()
