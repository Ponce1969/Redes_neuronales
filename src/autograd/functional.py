"""
API funcional estilo PyTorch para autograd Fase 6.
Operaciones de red completas usando Value.
"""

from __future__ import annotations
from typing import List
from .value import Value


def linear(inputs: List[Value], weights: List[Value], bias: Value) -> Value:
    """
    Capa lineal: y = Σ(x_i * w_i) + b
    """
    out = bias
    for x, w in zip(inputs, weights):
        out = out + x * w
    return out


def mse_loss(y_pred: List[Value], y_true: List[float]) -> Value:
    """
    Mean Squared Error: (1/n) * Σ(pred - true)²
    """
    losses = [(yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)]
    n = len(losses)
    
    if n == 0:
        return Value(0.0)
    
    total = losses[0]
    for l in losses[1:]:
        total = total + l
    
    return total * (1.0 / n)


def cross_entropy_loss(y_pred: List[Value], y_true: List[int]) -> Value:
    """
    Cross-entropy para clasificación
    """
    assert len(y_pred) == len(y_true)
    
    losses = []
    for i, (yp, yt) in enumerate(zip(y_pred, y_true)):
        # Softmax + cross-entropy simplificado
        exp_sum = sum([p.exp() for p in y_pred])
        loss = -yp.exp() / exp_sum
        losses.append(loss)
    
    return sum(losses) * (-1.0 / len(losses))


def softmax(values: List[Value]) -> List[Value]:
    """Softmax sobre lista de valores"""
    exp_values = [v.exp() for v in values]
    exp_sum = sum(exp_values)
    return [exp_val / exp_sum for exp_val in exp_values]
