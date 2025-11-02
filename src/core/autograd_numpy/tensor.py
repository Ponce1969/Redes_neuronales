from __future__ import annotations

import numpy as np


class Tensor:
    """
    Tensor vectorizado con operaciones diferenciables básicas.
    Compatible con el motor cognitivo y con NumPy nativo.
    """

    def __init__(self, data: np.ndarray | float | list, requires_grad: bool = False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def __add__(self, other: Tensor | float) -> Tensor:
        return Tensor(self.data + (other.data if isinstance(other, Tensor) else other))

    def __mul__(self, other: Tensor | float) -> Tensor:
        return Tensor(self.data * (other.data if isinstance(other, Tensor) else other))

    def matmul(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data)

    def tanh(self) -> Tensor:
        return Tensor(np.tanh(self.data))

    def sigmoid(self) -> Tensor:
        return Tensor(1 / (1 + np.exp(-self.data)))

    def relu(self) -> Tensor:
        return Tensor(np.maximum(0, self.data))

    def detach(self) -> Tensor:
        """Desconecta del grafo (simboliza un paso de recursión profunda tipo TRM)."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"
