import logging
from typing import Generator, Tuple
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    LinearLR,
)

from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.runner import Runner
from ds.tracking import Stage
from ds.models import THGStrainStressCNN
from ds.loss import weighted_l1_loss


class CrossRunner:
    def __init__(self, kf, groups, cfg: THGStrainStressConfig):
        self.kf = kf
        self.groups = groups
        self.cfg = cfg

    def __call__(
        self,
        dataset,
        local_rank: int,
        world_size: int,
    ) -> Generator[Tuple[Runner, Runner], None, None]:
        for train_idx, val_idx in self.kf.split(
            X=np.zeros(len(dataset)),
            y=self.groups,
            groups=self.groups,
        ):
            logging.debug(
                f"Training sets: {self.groups[train_idx]}\n"
                f"Validation sets: {self.groups[val_idx]}"
            )

            # Build the model with hparams defined by Optuna.
            # Use SyncBatchNorm to sync statistics between parallel models.
            # Cast the model to a DistributedDataParallel model where every GPU
            # has the full model.
            model = THGStrainStressCNN(self.cfg)
            model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

            # Choose optimizer with learning rates picked by Optuna.
            loss_fn = weighted_l1_loss  # MAE.
            optimizer = getattr(torch.optim, self.cfg.params.optimizer)(
                model.parameters(),
                lr=self.cfg.params.optimizer.lr,
                weight_decay=self.cfg.params.optimizer.weight_decay,
            )

            # Used for automatic mixed precision (AMP) to prevent underflow.
            scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

            # Stabilize the initial model.
            warmup_scheduler = LinearLR(
                optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=30
            )

            # Cosine annealing is used as a way to get to local minima.
            # Warm restarts are used to try and find neighbouring local minima.
            # https://arxiv.org/abs/1608.03983
            restart_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=self.cfg.params.scheduler.T_0,
                T_mult=self.cfg.params.scheduler.T_mult,
            )

            scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Distributed the workload across the GPUs.
            train_sampler = DistributedSampler(train_subset)
            val_sampler = DistributedSampler(val_subset)

            # Define dataloaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.cfg.params.batch_size,
                num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False,  # The distributed sampler shuffles for us.
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.cfg.params.batch_size,
                num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
                pin_memory=True,
                persistent_workers=True,
                shuffle=False,
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

            yield train_runner, val_runner

    @classmethod
    def StratifiedKFold(cls, n_splits, groups, cfg: THGStrainStressConfig):
        skf = StratifiedKFold(n_splits=n_splits, random_state=cfg.seed)
        return cls(skf, groups, cfg)
