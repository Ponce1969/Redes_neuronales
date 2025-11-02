from __future__ import annotations

from typing import Iterable, List

import numpy as np

try:
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
except ModuleNotFoundError:  # ejecución con PYTHONPATH=src
    from core.autograd_numpy.tensor import Tensor  # type: ignore

try:
    from src.autograd.value import Value  # type: ignore
except ModuleNotFoundError:
    from autograd.value import Value  # type: ignore

Parameter = Tensor | Value


def _zeros_like(param: Parameter) -> np.ndarray:
    data = param.data
    return np.zeros_like(data, dtype=np.float32)


def _to_numpy(array_like) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float32)


class SGD:
    """Stochastic Gradient Descent básico para Tensor y Value."""

    def __init__(self, params: Iterable[Parameter], lr: float = 0.01):
        self.params: List[Parameter] = list(params)
        self.lr = lr

    def step(self, grads: Iterable[np.ndarray]) -> None:
        for param, grad in zip(self.params, grads):
            g = _to_numpy(grad)
            if isinstance(param, Tensor):
                param.data -= self.lr * g
            else:
                param.data -= float(self.lr * g)


class Adam:
    """Versión ligera de Adam (soporta Tensor y Value)."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params: List[Parameter] = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [_zeros_like(p) for p in self.params]
        self.v = [_zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: Iterable[np.ndarray]) -> None:
        self.t += 1
        for idx, (param, grad) in enumerate(zip(self.params, grads)):
            g = _to_numpy(grad)

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if isinstance(param, Tensor):
                param.data -= update
            else:
                param.data -= float(update)
