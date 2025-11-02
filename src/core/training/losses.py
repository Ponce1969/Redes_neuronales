from __future__ import annotations

import numpy as np

try:
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
except ModuleNotFoundError:
    from core.autograd_numpy.tensor import Tensor  # type: ignore


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.data - target.data
    val = np.mean(diff ** 2)
    return Tensor(val)


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    val = np.mean(np.abs(pred.data - target.data))
    return Tensor(val)


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    eps = 1e-8
    p = np.clip(pred.data, eps, 1 - eps)
    val = -np.mean(target.data * np.log(p) + (1 - target.data) * np.log(1 - p))
    return Tensor(val)
