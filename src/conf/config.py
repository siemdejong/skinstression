from dataclasses import dataclass


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
    log: str
    data: str
    targets: str


@dataclass
class Params:
    epoch_count: int
    batch_size: int
    k_folds: int
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler


@dataclass
class THGStrainStressConfig:
    paths: Paths
    params: Params
