from __future__ import annotations

import numpy as np

from .autograd_numpy.tensor import Tensor


class TRMBlock:
    """
    Tiny Recursive Model (TRM) inspirado en Meta 2025.
    Contiene un estado latente z que evoluciona en pasos de razonamiento.
    """

    def __init__(self, n_in: int, n_hidden: int = 8, n_steps: int = 3):
        self.W_in = Tensor(np.random.randn(n_in, n_hidden) * 0.1)
        self.W_z = Tensor(np.random.randn(n_hidden, n_hidden) * 0.1)
        self.W_out = Tensor(np.random.randn(n_hidden, 1) * 0.1)
        self.b = Tensor(np.zeros((1, n_hidden)))
        self.z = Tensor(np.zeros((1, n_hidden)))
        self.n_steps = n_steps

    def step(self, x: Tensor) -> Tensor:
        """Un solo paso de razonamiento recursivo."""
        z_new = Tensor(
            np.tanh(x.data @ self.W_in.data + self.z.data @ self.W_z.data + self.b.data)
        )
        self.z = z_new.detach()
        y = Tensor(np.tanh(self.z.data @ self.W_out.data))
        return y

    def forward(self, x: Tensor) -> Tensor:
        out = None
        for _ in range(self.n_steps):
            out = self.step(x)
        assert out is not None
        return out

    def zero_state(self) -> None:
        self.z = Tensor(np.zeros_like(self.z.data))
