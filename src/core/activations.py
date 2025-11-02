"""
Módulo de funciones de activación.
Todas las funciones devuelven tanto la activación como su derivada
(para uso futuro en backpropagation).
"""

from __future__ import annotations
import math
from typing import Callable


class Activation:
    """Agrupa función de activación y su derivada."""

    def __init__(self, func: Callable[[float], float], deriv: Callable[[float], float], name: str):
        self.func = func
        self.deriv = deriv
        self.name = name

    def __call__(self, x: float) -> float:
        return self.func(x)

    def derivative(self, x: float) -> float:
        return self.deriv(x)

    def __repr__(self) -> str:
        return f"<Activation {self.name}>"


# === Activaciones clásicas === #

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: float) -> float:
    return max(0.0, x)

def relu_deriv(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    return math.tanh(x)

def tanh_deriv(x: float) -> float:
    return 1.0 - math.tanh(x) ** 2

def linear(x: float) -> float:
    return x

def linear_deriv(_: float) -> float:
    return 1.0


# === Instancias predefinidas === #
SIGMOID = Activation(sigmoid, sigmoid_deriv, "sigmoid")
RELU = Activation(relu, relu_deriv, "relu")
TANH = Activation(tanh, tanh_deriv, "tanh")
LINEAR = Activation(linear, linear_deriv, "linear")

# Mapeo rápido
ACTIVATIONS = {
    "sigmoid": SIGMOID,
    "relu": RELU,
    "tanh": TANH,
    "linear": LINEAR,
}
