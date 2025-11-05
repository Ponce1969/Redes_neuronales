"""Configuración base para el planificador cognitivo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SchedulerConfig:
    """Intervalos (en segundos) y banderas para el ciclo autónomo."""

    train_interval: int = 1800
    save_interval: int = 3600
    federation_interval: int = 7200
    evolution_interval: int = 10_800
    sleep_interval: int = 21_600

    enable_federation: bool = True
    enable_evolution: bool = True
    enable_sleep: bool = True

    loop_sleep: float = 10.0


__all__ = ["SchedulerConfig"]
