from dataclasses import dataclass
from typing import TypeVar

import numpy as np


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


@dataclass
class DecayParams:
    base: float = 1e-3
    decay_rate: float = 0.1
    decay_steps: float = 1_000_000


ParamType = TypeVar("ParamType")


