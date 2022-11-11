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
from typing import Any, Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    LinearLR,
)
from torch.utils.data import Subset, DataLoader
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
import numpy as np

from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.models import THGStrainStressCNN
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment
from ds.loss import weighted_l1_loss
from ds.utils import seed_all, ddp_cleanup, ddp_setup
from ds.logging_setup import setup_worker_logging
from ds.cross_validation import CrossRunner


class Trainer:
    def __init__(self, dataset_train, dataset_val, groups, cfg: THGStrainStressConfig):
        self.cfg = cfg
        if self.cfg.try_overfit:
            self.train_subset = Subset(dataset_train, indices=[0, 1])
            self.val_subset = Subset(dataset_val, indices=[0, 1])
        else:
            # Split the dataset in train, validation and test (sub)sets.
            train_val_size = int(len(dataset_train) * 0.8)
            train_val_idx, test_idx = train_test_split(
                np.arange(len(dataset_train)),
                train_size=train_val_size,
                stratify=groups,
                shuffle=True,
                random_state=cfg.seed,
            )

            train_size = int(len(train_val_idx) * 0.8)
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_idx)),
                train_size=train_size,
                stratify=groups[train_val_idx],
                shuffle=True,
                random_state=cfg.seed,
            )

            logging.debug(f"train idx: {train_idx}")
            logging.debug(f"val idx: {val_idx}")
            logging.debug(f"test idx: {test_idx}")

            # TODO: CROSSRUNNERS NOW ONLY USE NON-AUGMENTED DATA!
            self.train_val_subset = Subset(dataset_val, indices=train_val_idx)
            self.train_subset = Subset(dataset_train, indices=train_idx)
            self.val_subset = Subset(dataset_val, indices=val_idx)
            # dataset_val has the same augmentations as the testset
            self.test_subset = Subset(dataset_val, indices=test_idx)

    def __call__(
        self,
        local_rank: int,
        world_size: int,
        cross_runner: Optional[CrossRunner] = None,
        from_checkpoint: bool = False,
    ):
        """
        Args:
            cfg: hydra configuration object.
        """
        if cross_runner:
            train_runner, val_runner = cross_runner(
                self.train_val_subset, local_rank, world_size
            )
        else:
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
                T_0=self.cfg.params.scheduler.T_0,
                T_mult=self.cfg.params.scheduler.T_mult,
            )
            scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])
            scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

            # Distributed the workload across the GPUs.
            train_sampler = DistributedSampler(self.train_subset, seed=self.cfg.seed)
            val_sampler = DistributedSampler(self.val_subset, seed=self.cfg.seed)

            # Define dataloaders
            train_loader = DataLoader(
                self.train_subset,
                batch_size=int(self.cfg.params.batch_size),
                num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False,  # The distributed sampler shuffles for us.
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                self.val_subset,
                batch_size=int(self.cfg.params.batch_size),
                num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False,  # The distributed sampler shuffles for us.
                sampler=val_sampler,
            )

            # Create the runners
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

        if from_checkpoint:
            train_runner.load_checkpoint(self.cfg.paths.checkpoint)
            val_runner.load_checkpoint(self.cfg.paths.checkpoint)

        # Setup the experiment tracker
        log_dir = os.getcwd() + "/tensorboard"
        tracker = TensorboardExperiment(log_path=log_dir)

        # Run epochs.
        for epoch_id in range(self.cfg.params.epoch_count):
            tracker.add_epoch_param("lr", scheduler.get_last_lr()[0], epoch_id)

            train_runner.loader.sampler.set_epoch(epoch_id)
            val_runner.loader.sampler.set_epoch(epoch_id)
            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                experiment=tracker,
                epoch_id=epoch_id,
                local_rank=local_rank,
            )

            if local_rank == 0:
                train_loss = train_runner.avg_loss
                val_loss = val_runner.avg_loss

                logging.info(f"epoch: {epoch_id} | train loss: {train_loss}")
                logging.info(f"epoch: {epoch_id} | validation loss: {val_loss}")

            train_runner.reset()
            val_runner.reset()

            tracker.flush()

    def test(self, local_rank: int, world_size: int):
        model = THGStrainStressCNN(self.cfg)
        model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        test_loader = DataLoader(
            self.test_subset,
            batch_size=int(self.cfg.params.batch_size),
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
            pin_memory=True,
        )
        test_runner = Runner(
            loader=test_loader,
            model=model,
            loss_fn=weighted_l1_loss,
            stage=Stage.TEST,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
        )

        test_runner.load_checkpoint(self.cfg.paths.checkpoint)


def train(
    local_rank: int,
    world_size: int,
    cfg: THGStrainStressConfig,
    log_queue: Queue,
    cross_validation: bool = False,
):

    global_rank = int(os.environ["SLURM_RANK"]) * cfg.dist.gpus_per_node + local_rank

    # Setup logging.
    setup_worker_logging(local_rank, log_queue, cfg.debug)

    # Set and seed device
    torch.cuda.set_device(local_rank)
    logging.info(f"Hello from device {global_rank} of {world_size}")
    seed_all(cfg.seed)

    # Initialize process group
    ddp_setup(global_rank, world_size)
    # Load datasets.
    # Careful: dataset_train and _val contain the same data,
    # but the augmentations are diffent.
    # Data still needs to be randomly split by Trainer.
    dataset_train, groups = THGStrainStressDataset.load_data(
        split="train",
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
        reweight="sqrt_inv",
        lds=True,
    )

    dataset_val, _ = THGStrainStressDataset.load_data(
        split="validation",
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
        reweight="sqrt_inv",
        lds=True,
    )

    trainer = Trainer(dataset_train, dataset_val, groups, cfg)
    if cross_validation:
        cross_runner = CrossRunner.StratifiedKFold(n_splits=5, groups=groups, cfg=cfg)
    else:
        cross_runner = None
    trainer(
        local_rank,
        world_size,
        cross_runner=cross_runner,
        from_checkpoint=cfg.load_checkpoint,
    )

    # Exit subprocesses.
    ddp_cleanup()
