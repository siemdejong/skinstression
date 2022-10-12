import os
import logging
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore

import torch
from torch.utils.data import ConcatDataset
from torchvision.transforms import RandomCrop, Resize, ToTensor, Grayscale, Compose
import pandas as pd
import numpy as np

# from ds.dataset import create_dataloader
from ds.models import THGStrainStressCNN
from ds.runner import run_fold, run_test
from ds.tensorboard import TensorboardExperiment
from ds.tracking import Stage
from sklearn.model_selection import RandomizedSearchCV
from ds.cross_validator import k_fold
from ds.dataset import THGStrainStressDataset
from conf.config import THGStrainStressConfig

cs = ConfigStore.instance()
cs.store(name="thg_strain_stress_config", node=THGStrainStressConfig)

log = logging.getLogger(__name__)

# For reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: THGStrainStressConfig) -> None:

    # Hydra creates an output directory automatically at cwd.
    # Use it for tensorboard summaries.
    log_dir = os.getcwd() + "/tensorboard"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using device {device}.")
    else:
        device = torch.device("cpu")
        log.warning(f"Using device {device}. Is GPU set up properly?")

    model = THGStrainStressCNN(
        cfg.params.model.dropout, cfg.params.model.num_output_features
    ).to(device)

    # TODO: Which loss function? KL/MSE/MAE/MSLE?
    # Maybe we can even use loss as a means of how accurate the sigmoid is.
    loss_fn = torch.nn.L1Loss().to(device)  # MAE.
    # loss_fn = torch.nn.MSELoss().to(device)

    if cfg.params.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.params.optimizer.lr,
            betas=(cfg.params.optimizer.beta_1, cfg.params.optimizer.beta_2),
        )
    elif cfg.params.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=cfg.params.optimizer.lr,
        )

    # TODO: Which LR scheduler to use?
    # CosineAnnealingLR, CyclicLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg.params.scheduler.T_0
    )

    dataset, groups = THGStrainStressDataset.load_data()

    log.info(f"Training on {len(dataset)} samples.")

    # BUG: USES OPTIMIZER AND MODEL OF PREVIOUS K-FOLD.
    runner_iter = k_fold(
        n_splits=cfg.params.k_folds,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        batch_size=cfg.params.batch_size,
        dataset=dataset,
        groups=groups,
    )

    for fold_id, (train_runner, val_runner) in enumerate(runner_iter):

        # Setup the experiment tracker
        tracker = TensorboardExperiment(log_path=log_dir)

        run_fold(
            train_runner=train_runner,
            val_runner=val_runner,
            experiment=tracker,
            scheduler=scheduler,
            fold_id=fold_id,
            epoch_count=cfg.params.epoch_count,
        )


if __name__ == "__main__":
    main()
