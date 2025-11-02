"""
Funciones de pérdida y sus derivadas (gradientes respecto a la salida).
"""

from __future__ import annotations
from typing import List
import math

# ---- MSE (Mean Squared Error) ----
def mse_loss(y_pred: List[float], y_true: List[float]) -> float:
    """
    MSE = (1/n) * sum_i (y_pred_i - y_true_i)^2
    """
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n

def mse_grad(y_pred: List[float], y_true: List[float]) -> List[float]:
    """
    dL/dy_pred_i = 2*(y_pred_i - y_true_i) / n
    """
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    return [2.0 * (p - t) / n for p, t in zip(y_pred, y_true)]


# ---- Binary Cross Entropy (para salidas en (0,1)) ----
def bce_loss(y_pred: List[float], y_true: List[float], eps: float = 1e-12) -> float:
    """
    Binary cross-entropy (media por elemento):
    -1/n * sum ( t * log(p) + (1-t) * log(1-p) )
    """
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    loss = 0.0
    for p, t in zip(y_pred, y_true):
        p = max(min(p, 1.0 - eps), eps)
        loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
    return loss / n

def bce_grad(y_pred: List[float], y_true: List[float], eps: float = 1e-12) -> List[float]:
    """
    Derivada dL/dp para BCE:
    d/dp [ -t log p - (1-t) log(1-p) ] = -(t/p) + (1-t)/(1-p)
    normalizado por n
    """
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    grads: List[float] = []
    for p, t in zip(y_pred, y_true):
        p = max(min(p, 1.0 - eps), eps)
        grads.append((-(t / p) + (1 - t) / (1 - p)) / n)
    return grads
