from __future__ import annotations

import numpy as np

from .autograd_numpy.tensor import Tensor
from .autograd_numpy.loss import mse_loss


class TRM_ACT_Block:
    """
    Tiny Recursive Model con Adaptive Computation Time (ACT).
    Introduce supervisión profunda y una unidad de parada aprendida.
    """

    def __init__(self, n_in: int, n_hidden: int = 8, max_steps: int = 6):
        self.W_in = Tensor(np.random.randn(n_in, n_hidden) * 0.1)
        self.W_z = Tensor(np.random.randn(n_hidden, n_hidden) * 0.1)
        self.W_out = Tensor(np.random.randn(n_hidden, 1) * 0.1)
        self.W_halt = Tensor(np.random.randn(n_hidden, 1) * 0.1)
        self.b = Tensor(np.zeros((1, n_hidden)))
        self.z = Tensor(np.zeros((1, n_hidden)))
        self.max_steps = max_steps

    def step(self, x: Tensor) -> tuple[Tensor, float]:
        """Ejecuta un paso de razonamiento recursivo."""
        z_pre = x.data @ self.W_in.data + self.z.data @ self.W_z.data + self.b.data
        z_new = Tensor(np.tanh(z_pre))
        y = Tensor(np.tanh(z_new.data @ self.W_out.data))
        h_raw = z_new.data @ self.W_halt.data
        h = 1 / (1 + np.exp(-h_raw))
        self.z = z_new.detach()
        return y, float(h.squeeze())

    def forward(self, x: Tensor) -> Tensor:
        """Combina resultados ponderados por probabilidades de parada."""
        self.z = Tensor(np.zeros_like(self.z.data))
        total_output = np.zeros((1, 1), dtype=np.float32)
        remaining_prob = 1.0

        for _ in range(self.max_steps):
            y_t, h_t = self.step(x)
            p_t = remaining_prob * h_t
            total_output += p_t * y_t.data
            remaining_prob *= (1.0 - h_t)
            if h_t > 0.5:
                break

        return Tensor(total_output)

    def deep_supervision_loss(self, x: Tensor, y_true: Tensor) -> Tensor:
        """Promedia la pérdida sobre cada paso de razonamiento."""
        self.z = Tensor(np.zeros_like(self.z.data))
        total_loss = 0.0
        for _ in range(self.max_steps):
            y_pred, _ = self.step(x)
            loss = mse_loss(y_pred, y_true)
            total_loss += float(loss.data)
        return Tensor(total_loss / self.max_steps)

    def zero_state(self) -> None:
        self.z = Tensor(np.zeros_like(self.z.data))
