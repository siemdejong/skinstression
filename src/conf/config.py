from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class Model:
    dropout: float
    num_output_features: int


@dataclass
class Optimizer:
    name: str
    lr: float
    beta_1: float
    beta_2: float


@dataclass
class Scheduler:
    T_0: int


@dataclass
class Paths:
    data: str
    targets: str
    optuna_db: str


@dataclass
class Params:
    epoch_count: int
    batch_size: int
    k_folds: int
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler


@dataclass
class Dist:
    cpus: int
    gpus: int


class Mode(Enum):
    TUNE: int = 0
    VISUALIZE: int = 1


@dataclass
class Optuna:
    study_name: str
    trials: int


@dataclass
class THGStrainStressConfig:
    paths: Paths
    params: Params
    dist: Dist
    mode: Mode
    optuna: Optuna
