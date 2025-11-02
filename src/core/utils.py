"""
Funciones auxiliares: inicialización de pesos, normalización, etc.
"""

from __future__ import annotations
import random
from typing import List


def init_weights(n: int, min_val: float = -1.0, max_val: float = 1.0) -> List[float]:
    """Genera una lista de pesos aleatorios entre min_val y max_val."""
    return [random.uniform(min_val, max_val) for _ in range(n)]


def normalize_vector(v: List[float]) -> List[float]:
    """Normaliza un vector para que su magnitud total sea 1."""
    magnitude = sum(x ** 2 for x in v) ** 0.5
    return [x / magnitude for x in v] if magnitude > 0 else v
