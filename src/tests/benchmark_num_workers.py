"""Benchmark number of dataloader workers.
Adapted from https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
"""

from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch
import os
from ds.dataset import THGStrainStressDataset
import logging

log = logging.getLogger(__name__)


def benchmark_num_workers():
    dataset_train, _ = THGStrainStressDataset.load_data(
        split="train",
        data_path="/scistor/guest/sjg203/projects/shg-strain-stress/data/preprocessed/z-stacks/",
        targets_path="/scistor/guest/sjg203/projects/shg-strain-stress/data/z-stacks/logistic_targets.csv",
        reweight="sqrt_inv",
        lds=True,
    )

    log.info(f"Access to {torch.cuda.get_device_name()}.")

    if int(os.environ.get("SLURM_CPUS_PER_GPU")) != mp.cpu_count():
        log.error("SLURM says something different than Torch.")
        log.info(f"SLURM {int(os.environ.get('SLURM_CPUS_PER_GPU'))}")
        log.info(f"Torch {mp.cpu_count()}")
        log.info("Check which variable you need. Using SLURM as maximum workers.")

    # For every dataloader, assign a multiple workers.
    for num_workers in range(2, int(os.environ.get("SLURM_CPUS_PER_GPU")), 1):

        # Build dataloader with variable number of workers
        train_loader = DataLoader(
            dataset_train,
            batch_size=16,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,  # Normally, the distributed sampler shuffles for us.
        )

        start = time()
        for _ in range(1, 3):
            for _, _ in enumerate(train_loader):
                pass
        end = time()
        log.info(f"Finished in {end - start:.5f} seconds with {num_workers} workers.")
