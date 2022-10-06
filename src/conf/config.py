from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    data: str
    targets: str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int
    dropout: float
    num_output_features: int
    beta_1: float
    beta_2: float


@dataclass
class THGStrainStressConfig:
    paths: Paths
    params: Params
