import pathlib

import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import numpy as np

import torch.distributed as dist
import logging as log


def create_experiment_log_dir(root: str, parents: bool = True) -> str:
    root_path = pathlib.Path(root).resolve()
    child = (
        create_from_missing(root_path)
        if not root_path.exists()
        else create_from_existing(root_path)
    )
    try:
        child.mkdir(parents=parents)
    except FileExistsError:
        log.info("Using existing log dir for this process.")
    return child.as_posix()


def create_from_missing(root: pathlib.Path) -> pathlib.Path:
    return root / "0"


def create_from_existing(root: pathlib.Path) -> pathlib.Path:
    children = [
        int(c.name) for c in root.glob("*") if (c.is_dir() and c.name.isnumeric())
    ]
    if is_first_experiment(children):
        child = create_from_missing(root)
    else:
        child = root / increment_experiment_number(children)
    return child


def is_first_experiment(children: list[int]) -> bool:
    return len(children) == 0


def increment_experiment_number(children: list[int]) -> str:
    return str(max(children) + 1)


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        )

    return kernel_window


def sturge(n):
    """Sturge's formula to calculate the number of bins a histogram could have
    assuming a normal distribution.
    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    k = int(np.ceil(np.log2(n)) + 1)
    return k


def ddp_setup(rank, world_size):
    """Setup a process group at a specific rank, being world-aware.
    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
    """
    # initialize the process group
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world_size
    )
    log.info(f"Process group initialized on rank {rank} of {world_size}.")


def ddp_cleanup():
    """Destroy the current process group."""
    dist.destroy_process_group()


def seed_all(seed=42):
    """Seed the current device.
    Used to make the optimalization determinant.
    However, not 100% determinant: https://pytorch.org/docs/stable/notes/randomness.html.
    Args:
        seed: seed to set the current device to.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
