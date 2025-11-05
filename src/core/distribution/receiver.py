"""Aplicación de memorias y pesos recibidos desde nodos remotos."""

from __future__ import annotations

import base64
from typing import Any, Iterable

import numpy as np

from core.persistence.serializer import load_memory, load_weights
from core.persistence.file_manager import memory_path, weight_path
from core.society.agent import CognitiveAgent


def apply_shared_memory(agent: CognitiveAgent, episodes: Iterable[dict[str, Any]]) -> int:
    """Incorpora episodios recibidos al agente destino."""

    count = 0
    for episode in episodes:
        agent.memory_system.memory.store(
            np.array(episode.get("input")),
            _maybe_array(episode.get("target")),
            np.array(episode.get("output")),
            float(episode.get("loss", 0.0)),
            episode.get("attention", {}),
        )
        count += 1
    return count


def apply_shared_weights(agent: CognitiveAgent, encoded_weights: str | None) -> bool:
    if not encoded_weights:
        return False

    target_path = weight_path(agent.name)
    target_path.write_bytes(base64.b64decode(encoded_weights))
    load_weights(agent)
    return True


def apply_shared_payload(agent: CognitiveAgent, payload: dict[str, Any]) -> dict[str, Any]:
    received = apply_shared_memory(agent, payload.get("memory", []))
    updated = apply_shared_weights(agent, payload.get("weights"))
    return {"received": received, "weights_applied": updated}


def _maybe_array(value: Any) -> Any:
    if value is None:
        return None
    return np.array(value)
