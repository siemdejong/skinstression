"""Provides utility functions like directory/file creation, convolution windows,
and function to facilitate parallel processing.
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

This file incorporates work covered by the following copyright and permission notice:  

    Copyright (c) 2021 Mark Todisco & ArjanCodes
    Copyright (c) 2021 Yuzhe Yang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import logging
import os
import pathlib
import socket

import numpy as np
import torch
import torch.distributed as dist
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


def create_experiment_log_dir(root: str, parents: bool = True) -> str:
    root_path = pathlib.Path(root).resolve()
    child = (
        create_from_missing(root_path)
        if not root_path.exists()
        else create_from_existing(root_path)
    )
    try:
        child.mkdir(parents=parents)
        logging.info("Tensorboard logging directory created.")
    except FileExistsError:
        logging.info("Using existing tensorboard log dir for this subprocess.")
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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    """Destroy the current process group."""
    dist.destroy_process_group()


def reduce_tensor(tensor):
    """When calling all_reduce(), tensors get reduced to their sum.
    However, it is more interesting to see what the average tensor is for the loss e.g.

    Args:
        tensor: tensor to be reduced.
    """
    cloned_tensor = tensor.clone()
    dist.all_reduce(cloned_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor = cloned_tensor / dist.get_world_size()

    return reduced_tensor


def get_ip():
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    return ip_addr


def get_free_port(host: str):
    sock = socket.socket()
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def seed_all(seed=42):
    """Seed the current device.
    Used to make the optimalization deterministic.
    However, not 100% deterministic: https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed: seed to set the current device to.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
