import os
import logging
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore

# from ds.dataset import create_dataloader
from ds.tensorboard import tensorboard_dev_upload
from ds.hyperparameters import tune_hyperparameters
from ds.visualization import visualize
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
    if cfg.tensorboard_dev.upload:
        tensorboard_dev_upload(cfg)

    if cfg.mode == Mode.TUNE.name:
        tune_hyperparameters(cfg)
    elif cfg.mode == Mode.VISUALIZE.name:
        visualize(cfg)


if __name__ == "__main__":
    main()
