from __future__ import annotations

import numpy as np

from .tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.data - target.data
    loss_val = np.mean(diff ** 2)
    return Tensor(loss_val)


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    eps = 1e-8
    pred_clipped = np.clip(pred.data, eps, 1 - eps)
    loss = -np.mean(
        target.data * np.log(pred_clipped) + (1 - target.data) * np.log(1 - pred_clipped)
    )
    return Tensor(loss)
