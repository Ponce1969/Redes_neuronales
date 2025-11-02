"""
Operaciones matemáticas para autograd Fase 6.
Funciones de conveniencia sobre Value.
"""

from __future__ import annotations
from .value import Value
import math


def relu(x: Value) -> Value:
    """ReLU: max(0, x)"""
    out = Value(max(0, x.data), (x,), "relu")
    
    def _backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out


def leaky_relu(x: Value, alpha: float = 0.01) -> Value:
    """Leaky ReLU: x si x >= 0, alpha*x si x < 0"""
    out = Value(x.data if x.data >= 0 else alpha * x.data, (x,), "leaky_relu")
    
    def _backward():
        x.grad += (1.0 if x.data >= 0 else alpha) * out.grad
    out._backward = _backward
    return out


def exp(x: Value) -> Value:
    """Exponencial: e^x"""
    return x.exp()


def log(x: Value) -> Value:
    """Logaritmo natural"""
    out = Value(math.log(x.data), (x,), "log")
    
    def _backward():
        x.grad += (1.0 / x.data) * out.grad
    out._backward = _backward
    return out


def sum_values(values: list[Value]) -> Value:
    """Suma de valores"""
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total
