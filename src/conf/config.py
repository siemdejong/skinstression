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
    T_mult: int
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
    TRAIN: int = 2


@dataclass
class Pruner:
    min_resource: int
    reduction_factor: int


@dataclass
class Hparams:
    optimizer_name: str
    weight_decay: list[float, float]
    lr: list[float, float]
    T_mult: list[float, float]
    num_preblocks: int
    dropout_1: list[float, float]
    dropout_2: list[float, float]
    dropout_3: list[float, float]
    dropout_4: list[float, float]
    n_nodes: int
    batch_size: list[int]


# class Direction(Enum):
#     minimize: int = 0
#     maximize: int = 1


@dataclass
class Optuna:
    study_name: str
    trials: int
    direction: str
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
    optuna: Optuna
    use_amp: bool
    seed: int
