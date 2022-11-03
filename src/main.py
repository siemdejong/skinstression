import hydra
from hydra.core.config_store import ConfigStore
from ds.utils import get_ip, get_free_port
from ds.hyperparameters import tune_hyperparameters
from ds.visualization import visualize
from ds.training import train
from conf.config import THGStrainStressConfig, Mode
import torch.multiprocessing as mp
from ds.logging_setup import setup_primary_logging
import os
import torch
import logging

cs = ConfigStore.instance()
cs.store(name="thg_strain_stress_config", node=THGStrainStressConfig)


# @hydra.main(config_path="conf", config_name="config")
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: THGStrainStressConfig) -> None:
    """
    This is the main entry point for the THG strain stress project.
    Can
    1. calculate appropriate hyperparameters for a model.
       To be instantiated with `sbatch shg-optuna.sbatch` as it makes
       use of Pytorch DistributedDataParallel and allows for multiple
       nodes using their own GPUs to divide the data.
    2. visualize result of hyperparameter optimization.
    3. calculate model parameters for the model,
       estimating parameters describing the strain-stress
       curve of skin tissue from single SHG images;

    Configurations must be made in conf/config.yaml.
    """
    # Sources:
    #   https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/
    #   https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

    # Initialize the primary logging handlers. Worker processes may use
    # the `log_queue` to push their messages to the same log file.
    # Logging processes need to be initialized with the `spawn` method.
    torch.multiprocessing.set_start_method("spawn", force=True)
    log_queue = setup_primary_logging("out.log", "error.log", cfg.debug)

    # IP and port need to be available for dist.init_process_group().
    ip = get_ip()
    port = get_free_port(ip)
    os.environ["MASTER_ADDR"] = str(ip)
    os.environ["MASTER_PORT"] = str(port)
    logging.info(
        f"Processes will spawn from {ip}:{port}. "
        "See Hydra output for worker log messages."
    )

    if cfg.mode == Mode.VISUALIZE.name:
        visualize(cfg.paths.optuna_db)
    elif cfg.mode == Mode.TUNE.name:
        mp.spawn(
            fn=tune_hyperparameters,
            nprocs=cfg.dist.gpus_per_node,
            args=(cfg.dist.gpus_per_node, cfg, log_queue),
        )
    elif cfg.mode == Mode.TRAIN.name:
        pass

    logging.info("All processes exited without critical errors.")


if __name__ == "__main__":
    main()
