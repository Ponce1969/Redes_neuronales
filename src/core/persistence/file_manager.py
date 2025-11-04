"""Utilidades de archivos para la capa de persistencia cognitiva."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

BASE_DIR = Path("data/persistence")


def ensure_dirs() -> None:
    """Crea directorios requeridos para guardar pesos y memorias."""

    (BASE_DIR / "weights").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "memories").mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Retorna un timestamp compacto para nombres de archivo."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def weight_path(agent_name: str) -> Path:
    ensure_dirs()
    return BASE_DIR / "weights" / f"{agent_name}_weights.npz"


def memory_path(agent_name: str) -> Path:
    ensure_dirs()
    return BASE_DIR / "memories" / f"{agent_name}_memory.json"
