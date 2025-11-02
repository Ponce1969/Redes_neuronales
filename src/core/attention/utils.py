from __future__ import annotations

import numpy as np


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable numéricamente sobre un vector 1D."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom == 0.0:
        return np.ones_like(exp_vals) / exp_vals.size
    return exp_vals / denom


def ensure_2d(vector: np.ndarray) -> np.ndarray:
    """Asegura representación (1, dim) para vectores 1D."""
    if vector.ndim == 1:
        return vector.reshape(1, -1)
    return vector
