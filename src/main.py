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
    log_dir = os.getcwd() + "/tensorboard"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using device {device}.")
    else:
        device = torch.device("cpu")
        log.warning(f"Using device {device}. Is GPU set up properly?")

    model = THGStrainStressCNN(cfg.params.dropout, cfg.params.num_output_features).to(
        device
    )

    # TODO: Which loss function? KL/MSE/MAE/MSLE?
    loss_fn = torch.nn.L1Loss().to(device)  # MAE.
    # loss_fn = torch.nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.params.lr,
        betas=(cfg.params.beta_1, cfg.params.beta_2),
    )

    # CosineAnnealingLR, CyclicLR
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 300)

    data_transform = Compose(
        [
            # RandomCrop(size=(258, 258)),
            Resize((258, 258)),
            Grayscale(),
            # AugMix(),
            # RandAugment(num_ops=2),
            ToTensor(),
            # Lambda(lambda y: (y - y.mean()) / y.std()), # To normalize the image.
        ]
    )
    datasets = []
    groups = []
    for _, labels in pd.read_csv(cfg.paths.targets).iterrows():

        folder = int(labels["index"])
        targets = labels[["A", "h", "slope", "C"]].to_numpy(dtype=float)

        if not (Path(cfg.paths.data) / str(folder)).is_dir():
            log.info(
                f"{Path(cfg.paths.data) / str(folder)} will be excluded "
                f"because it is not provided in {cfg.paths.targets}"
            )
            continue

        dataset = THGStrainStressDataset(
            root_data_dir=cfg.paths.data,
            folder=folder,
            targets=targets,
            data_transform=data_transform,
        )
        datasets.append(dataset)
        groups.extend([folder] * len(dataset))

    groups = np.array(groups)
    dataset = ConcatDataset(datasets)

    runner_iter = k_fold(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        batch_size=cfg.params.batch_size,
        dataset=dataset,
        groups=groups,
    )

    # Setup the experiment tracker
    tracker = TensorboardExperiment(log_path=log_dir)

    for fold_id, (train_runner, val_runner) in enumerate(runner_iter):
        run_fold(
            train_runner=train_runner,
            val_runner=val_runner,
            experiment=tracker,
            # scheduler=scheduler,
            fold_id=fold_id,
            epoch_count=cfg.params.epoch_count,
        )


if __name__ == "__main__":
    main()
