import logging
import hydra
from hydra.core.config_store import ConfigStore
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
    """
    This is the main entry point for the THG strain stress project.
    Can
    1. calculate model parameters for a predefined convolutional
    neural network, estimating parameters describing the strain-stress
    curve of skin tissue from single SHG images;
    2. calculate appropriate hyperparameters for this model;
    3. visualize result of hyperparameter optimization.

    Configurations must be made in conf/config.yaml.
    """

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
