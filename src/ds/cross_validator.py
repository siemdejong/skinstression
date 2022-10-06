from sklearn.model_selection import KFold, StratifiedGroupKFold
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, SequentialSampler
from torchvision.transforms import Compose, Grayscale, RandomCrop, ToTensor
from ds.runner import Runner
import torch
from ds.tracking import Stage
from ds.dataset import THGStrainStressDataset
from typing import Generator, Tuple
import logging

log = logging.getLogger(__name__)


def k_fold(
    n_splits: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    dataset: Dataset,
    groups: np.ndarray,
) -> Generator[Tuple[Runner, Runner], None, None]:

    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    for train_idx, val_idx in sgkf.split(
        X=np.ones((len(dataset), 1)), y=groups, groups=groups
    ):

        log.debug(
            f"Training sets: {groups[train_idx]}\n"
            f"Validation sets: {groups[val_idx]}"
        )

        # Not using SubsetRandomSampler, as the sampler
        # might choose overlapping samples, thus not utilizing the whole dataset.
        # Train/validation sets are shuffled internatlly in the dataloader.
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=batch_size
        )
        val_loader = DataLoader(
            dataset=val_dataset, shuffle=True, batch_size=batch_size
        )

        # Create the runners
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            device=device,
        )
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            device=device,
        )

        yield train_runner, val_runner
