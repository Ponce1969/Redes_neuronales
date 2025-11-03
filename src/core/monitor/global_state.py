"""Estado global compartido entre demos y dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = PROJECT_ROOT / "dashboard_state.json"

try:
    _STATE_CACHE: dict[str, Any] = json.loads(STATE_FILE.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    _STATE_CACHE = {}

_GRAPH: Optional[Any] = None
_STATE_LOCK = Lock()


def set_graph(graph: Any) -> None:
    """Registra la instancia del grafo en memoria y actualiza snapshot."""

    global _GRAPH
    _GRAPH = graph
    record_graph_metadata(graph)


def get_graph() -> Optional[Any]:
    """Obtiene la instancia del grafo si está disponible en el intérprete."""

    return _GRAPH


def record_graph_metadata(graph: Any) -> None:
    write_state_snapshot("graph_metadata", _extract_graph_metadata(graph))


def record_monitor_state(state: dict[str, Any]) -> None:
    write_state_snapshot("monitor", state)


def record_memory_state(state: list[dict[str, Any]]) -> None:
    write_state_snapshot("memory", state)


def write_state_snapshot(section: str, payload: Any) -> None:
    """Escribe (o actualiza) una sección del snapshot JSON compartido."""

    with _STATE_LOCK:
        _STATE_CACHE[section] = payload
        tmp_path = STATE_FILE.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(_STATE_CACHE, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(STATE_FILE)


def load_state_snapshot() -> dict[str, Any]:
    """Carga el snapshot persistido desde disco (si existe)."""

    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_graph_metadata(graph: Any) -> dict[str, Any]:
    blocks = list(getattr(graph, "blocks", {}).keys())
    connections = {
        dest: sources for dest, sources in getattr(graph, "connections", {}).items()
    }
    return {
        "blocks": blocks,
        "connections": connections,
        "has_monitor": hasattr(graph, "monitor"),
        "has_memory_system": hasattr(graph, "memory_system"),
    }
