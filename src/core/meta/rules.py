"""Reglas adaptativas para meta-aprendizaje ligero."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def adaptive_lr(prev_loss: float, curr_loss: float, lr: float) -> float:
    """Ajusta el learning rate de forma heurística.

    - Si la pérdida aumenta respecto a la anterior, reduce el paso.
    - Si la pérdida mejora de forma apreciable, permite un leve incremento.
    - Mantiene el valor dentro del rango [1e-5, 1e-1].
    """

    delta = curr_loss - prev_loss
    if prev_loss == np.inf:
        return float(lr)

    if delta > 0.0:
        lr *= 0.7
    elif delta < -0.001:
        lr *= 1.05

    return float(np.clip(lr, 1e-5, 1e-1))


def adaptive_focus(att_mean: float, focus: float) -> float:
    """Ajusta el foco atencional global según la atención promedio."""

    if att_mean < 0.2:
        focus *= 1.2
    elif att_mean > 0.8:
        focus *= 0.9

    return float(np.clip(focus, 0.1, 2.0))


def adaptive_sleep(loss_trend: Iterable[float], base_interval: int) -> int:
    """Modula la frecuencia de consolidación (sleep cycles).

    Considera la tendencia reciente de la pérdida. Si sube, aumenta el
    intervalo (más consolidación). Si desciende de forma sostenida, permite
    dormir con menos frecuencia. El intervalo mínimo es 1.
    """

    history = list(loss_trend)
    if len(history) < 3:
        return base_interval

    slope = float(np.mean(np.diff(history[-3:])))
    interval = base_interval

    if slope > 0:
        interval += 1
    elif slope < -0.002:
        interval = max(1, interval - 1)

    return interval
