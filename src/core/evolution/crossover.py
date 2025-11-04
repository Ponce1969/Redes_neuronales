"""Utilidades de cruce y mutación para pesos del grafo cognitivo."""

from __future__ import annotations

import numpy as np


def crossover_weights(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Combina dos matrices de pesos aplicando crossover y mutación ligera."""

    if w1.shape != w2.shape:
        raise ValueError("Los tensores a cruzar deben tener la misma forma")

    mask = np.random.rand(*w1.shape) < 0.5
    child = np.where(mask, w1, w2)

    mutation = np.random.randn(*w1.shape) * 0.02
    return (child + mutation).astype(np.float32)


def crossover_scalar(a: float, b: float) -> float:
    """Cruza dos valores escalares reutilizando la lógica matricial."""

    arr_a = np.array([[a]], dtype=np.float32)
    arr_b = np.array([[b]], dtype=np.float32)
    return float(crossover_weights(arr_a, arr_b)[0, 0])
