from typing import Any
from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment
from ds.models import THGStrainStressCNN

from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingWarmRestarts,
)

import logging
from torch import nn
import torch
from torch.utils.data import random_split
import os

class Trainer:
    def __init__(self, dataset):
        # Split the dataset in train, validation and test (sub)sets.
        train_test_split = int(len(dataset) * 0.8)
        train_set, test_set = random_split(
            dataset, [train_test_split, len(dataset) - train_test_split]
        )

        train_val_split = int(len(train_set) * 0.8)
        self.train_subset, self.val_subset = random_split(
            train_set, [train_val_split, len(train_set) - train_val_split]
        )

    def __call__(self, cfg: THGStrainStressConfig):
        """
        Args:
            cfg: hydra configuration object.
        """
        use_cuda = torch.cuda.is_available()
        logging.info(f"CUDA available: {use_cuda}")
        device = torch.device("cuda" if use_cuda else "cpu")

        model = THGStrainStressCNN(cfg)

        model = model.to(device)
        loss_fn = nn.L1Loss()  # MAE.
        optimizer = getattr(torch.optim, cfg.params.optimizer.name)(
            model.parameters(),
            lr=cfg.params.optimizer.lr,
            weight_decay=cfg.params.optimizer.weight_decay,
        )
        warmup_scheduler = LinearLR(
            optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=10
        )
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=400,
            T_mult=cfg.params.optimizer.T_mult,
        )
        scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        # Define dataloaders
        train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=int(cfg.params.batch_size),
            num_workers=2,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=int(cfg.params.batch_size),
            num_workers=2,
            pin_memory=True,
        )

        # Create the runners
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            device=device,
            progress_bar=True,
            scaler=scaler,
        )
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            progress_bar=True,
            scaler=scaler,
        )

        # Setup the experiment tracker
        log_dir = os.getcwd() + "/tensorboard"
        tracker = TensorboardExperiment(log_path=log_dir)

        # Run epochs.
        for epoch_id in range(cfg.params.epoch_count):
            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                experiment=tracker,
                epoch_id=epoch_id,
            )

            loss = val_runner.avg_loss
            logging.info(f"epoch: {epoch_id} | loss: {loss}")


def train(cfg: THGStrainStressConfig):
    # Load dataset.
    dataset, groups = THGStrainStressDataset.load_data(
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
    )
    trainer = Trainer(dataset)
    trainer(cfg)
