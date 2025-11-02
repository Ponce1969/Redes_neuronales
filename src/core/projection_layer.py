from __future__ import annotations

import numpy as np

try:  # compatibilidad con importaciones desde el paquete "src"
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
except ModuleNotFoundError:  # cuando PYTHONPATH apunta directamente a src/
    from core.autograd_numpy.tensor import Tensor  # type: ignore


class ProjectionLayer:
    """Transformación lineal aprendible para alinear dimensiones entre bloques."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = Tensor(np.random.randn(input_dim, output_dim) * 0.1)
        self.b = Tensor(np.zeros((1, output_dim), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        x_data = x.data if isinstance(x, Tensor) else np.array(x, dtype=np.float32)
        return Tensor(x_data @ self.W.data + self.b.data)

    def __repr__(self) -> str:  # pragma: no cover - para depuración
        return f"ProjectionLayer({self.input_dim}->{self.output_dim})"
