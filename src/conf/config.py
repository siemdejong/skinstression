"""Configuration dataclasses to be read by Hydra.
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

from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class Model:
    dropout_1: float
    dropout_2: float
    dropout_3: float
    dropout_4: float
    n_nodes: 64
    num_output_features: int


@dataclass
class Optimizer:
    name: str
    lr: float
    weight_decay: float
    beta_1: float
    beta_2: float


@dataclass
class Scheduler:
    T_0: int
    T_mult: int


@dataclass
class Paths:
    data: str
    targets: str
    optuna_db: str
    checkpoint: str


@dataclass
class Params:
    epoch_count: int
    batch_size: int
    k_folds: int
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler
    loss_fn: str


@dataclass
class Dist:
    nodes: int
    gpus_per_node: int
    cpus_per_gpu: int


class Mode(Enum):
    TUNE: int = 0
    TUNE_VISUALIZE: int = 1
    TRAIN: int = 2
    CROSS_VALIDATION: int = 3


@dataclass
class Pruner:
    min_resource: int
    reduction_factor: int


@dataclass
class Hparams:
    optimizer_name: str
    weight_decay: list[float, float]
    lr: list[float, float]
    T_0: list[int, int]
    T_mult: list[int, int]
    num_preblocks: list[int]
    n_nodes: int
    batch_size: list[int]


# class Direction(Enum):
#     minimize: int = 0
#     maximize: int = 1


@dataclass
class Optuna:
    study_name: str
    trials: int
    direction: str  # Use Direction class?
    hparams: Hparams
    seed: int
    pruner: Pruner
    parallel: bool


@dataclass
class THGStrainStressConfig:
    paths: Paths
    params: Params
    dist: Dist
    mode: Mode
    debug: bool
    optuna: Optuna
    use_amp: bool
    seed: int
    try_overfit: bool
    dry_run: bool
    load_checkpoint: bool
