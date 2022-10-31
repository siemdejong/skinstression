import hydra
from hydra.core.config_store import ConfigStore
from ds.utils import seed_all
from ds.hyperparameters import tune_hyperparameters
from ds.visualization import visualize
from ds.training import train
from conf.config import THGStrainStressConfig, Mode
import torch.multiprocessing as mp
from ds.logging_setup import setup_primary_logging
from argparse import ArgumentParser
import os

cs = ConfigStore.instance()
cs.store(name="thg_strain_stress_config", node=THGStrainStressConfig)


# @hydra.main(config_path="conf", config_name="config")
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: THGStrainStressConfig) -> None:
    """
    This is the main entry point for the THG strain stress project.
    Can
    1. calculate appropriate hyperparameters for a model;
    2. visualize result of hyperparameter optimization.
    3. calculate model parameters for the model,
       estimating parameters describing the strain-stress
       curve of skin tissue from single SHG images;

    Configurations must be made in conf/config.yaml.
    """
    # Sources:
    #   https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/
    #   https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

    parser = ArgumentParser()
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument(
        "--ip_adress", type=str, required=True, help="ip address of the host node"
    )
    parser.add_argument("--ngpus", default=1, type=int, help="number of gpus per node")

    args = parser.parse_args()

    # Total number of GPUs availabe.
    args.world_size = args.ngpu * args.nodes

    # Add the ip address to the environment variable so it can be easily available.
    os.environ["MASTER_ADDR"] = args.ip_adress
    os.environ["MASTER_PORT"] = "8888"
    os.environ["WORLD_SIZE"] = str(args.world_size)

    if cfg.mode == Mode.VISUALIZE.name:
        fn = visualize
    elif cfg.mode == Mode.TUNE.name:
        fn = tune_hyperparameters
    elif cfg.mode == Mode.TRAIN.name:
        fn = train

    # Initialize the primary logging handlers. Use the returned `log_queue`
    # to which the worker processes would use to push their messages
    log_queue = setup_primary_logging("out.log", "error.log")

    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(fn, nprocs=args.ngpus, args=(args, cfg, log_queue))


if __name__ == "__main__":
    main()
