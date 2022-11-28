"""Provides main entry to the project.
Copyright (C) 2022  Siem de Jong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import os

import hydra
import torch
import torch.multiprocessing as mp
from hydra.core.config_store import ConfigStore

from conf.config import Mode, SkinstressionConfig
from ds.hyperparameters import tune_hyperparameters
from ds.logging_setup import setup_primary_logging
from ds.training import train
from ds.utils import get_free_port, get_ip
from ds.visualization import visualize
from tests.benchmark_num_workers import benchmark_num_workers

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="skinstression_config", node=SkinstressionConfig)


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: SkinstressionConfig) -> None:
    """
    This is the main entry point for the Skinstression project.
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
    #   https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide

    # Initialize the primary logging handlers. Worker processes may use
    # the `log_queue` to push their messages to the same log file.
    # Logging processes need to be initialized with the `spawn` method.
    torch.multiprocessing.set_start_method("spawn", force=True)
    log_queue = setup_primary_logging("out.log", "error.log", cfg.debug)

    if cfg.mode == Mode.TUNE_VISUALIZE.name:
        visualize(cfg)
    elif cfg.mode == Mode.TUNE.name:
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tune_hyperparameters(global_rank, local_rank, world_size, cfg, log_queue)
    elif cfg.mode == Mode.TRAIN.name:
        with torch.autograd.profiler.profile(
            enabled=cfg.profiler.enable,
            use_cuda=cfg.profiler.cuda,
            profile_memory=cfg.profiler.memory,
        ) as prof:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            train(global_rank, local_rank, world_size, cfg, log_queue)
        try:
            log.info(prof.key_averages().table(sort_by="self_cpu_time_total"))
        except AttributeError:
            pass
    elif cfg.mode == Mode.CROSS_VALIDATION.name:
        # world_size = cfg.dist.gpus_per_node * cfg.dist.nodes
        world_size = int(os.environ["SLURM_GPUS_PER_NODE"]) * int(
            os.environ["SLURM_JOB_NUM_NODES"]
        )
        mp.spawn(
            fn=train,
            # nprocs=cfg.dist.gpus_per_node,
            nprocs=int(os.environ["SLURM_GPUS_PER_NODE"]),
            args=(world_size, cfg, log_queue, True),
        )
    elif cfg.mode == Mode.BENCHMARK_NUM_WORKERS.name:
        benchmark_num_workers()

    log.info("All processes exited without critical errors.")


if __name__ == "__main__":
    main()
