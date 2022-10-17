import os
import logging
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore

from ray import tune
import torch
from torch import nn
from torch.utils.data import ConcatDataset, random_split
from torchvision.transforms import RandomCrop, Resize, ToTensor, Grayscale, Compose
import pandas as pd
import numpy as np


# from ds.dataset import create_dataloader
from ds.models import THGStrainStressCNN
from ds.runner import run_epoch, run_fold, run_test
from ds.runner import Runner
from ds.tensorboard import TensorboardExperiment
from ds.tracking import Stage
from ds.hyperparameters import tune_hyperparameters
from ds.visualization import visualize
from sklearn.model_selection import RandomizedSearchCV
from ds.cross_validator import k_fold
from ds.dataset import THGStrainStressDataset
from conf.config import THGStrainStressConfig, Mode

cs = ConfigStore.instance()
cs.store(name="thg_strain_stress_config", node=THGStrainStressConfig)

log = logging.getLogger(__name__)

# For reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: THGStrainStressConfig) -> None:

    # TODO: Which LR scheduler to use?
    # CosineAnnealingLR, CyclicLR
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, cfg.params.scheduler.T_0
    # )

    if cfg.mode == Mode.TUNE.name:
        tune_hyperparameters(cfg)
    elif cfg.mode == Mode.VISUALIZE.name:
        visualize(cfg)


if __name__ == "__main__":
    main()
