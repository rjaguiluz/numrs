
from typing import Literal, List, Any

OptimizerType = Literal[
    "sgd", "adam", "adamw", "nadam", "radam", "lamb", 
    "adabound", "rmsprop", "adagrad", "adadelta", "lbfgs", "rprop"
]

LossType = Literal["mse", "cross_entropy"]

class Dataset:
    def __init__(self, inputs: List[List[float]], targets: List[List[float]], batch_size: int): ...

class Trainer:
    def __init__(self, model: Any, optimizer: OptimizerType = "sgd", loss: LossType = "mse", lr: float = 0.01): ...
    def fit(self, dataset: Dataset, epochs: int) -> None: ...
