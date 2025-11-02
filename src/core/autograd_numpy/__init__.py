"""Lightweight NumPy-based autograd primitives used by TRM blocks."""

from .tensor import Tensor
from .loss import mse_loss, binary_cross_entropy

__all__ = ["Tensor", "mse_loss", "binary_cross_entropy"]
