try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

OptimizerType = Literal[
    "sgd", "adam", "adamw", "nadam", "radam", "lamb", 
    "adabound", "rmsprop", "adagrad", "adadelta", "lbfgs", "rprop"
]

LossType = Literal["mse", "cross_entropy"]

DType = Literal["float32"]

__all__ = ["OptimizerType", "LossType", "DType"]
