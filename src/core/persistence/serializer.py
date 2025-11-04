"""Serialización de pesos y memorias para agentes cognitivos."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

import numpy as np

from core.persistence.file_manager import memory_path, weight_path


def save_weights(agent: Any) -> None:
    """Guarda pesos relevantes de un agente cognitivo."""

    data: Dict[str, Dict[str, Any]] = {}
    for name, block in agent.graph.blocks.items():
        perceiver = getattr(block, "perceiver", None)
        if perceiver is None:
            continue

        perceiver_dict: Dict[str, Any] = {}

        input_weights = getattr(perceiver, "input_weights", None)
        if input_weights is not None:
            perceiver_dict["input_weights"] = [float(w.data) for w in input_weights]

        memory_weights = getattr(perceiver, "memory_weights", None)
        if memory_weights is not None:
            perceiver_dict["memory_weights"] = [float(w.data) for w in memory_weights]

        gate_weights = getattr(perceiver, "gate_weights", None)
        if gate_weights is not None:
            perceiver_dict["gate_weights"] = [float(w.data) for w in gate_weights]

        gate_bias = getattr(perceiver, "gate_bias", None)
        if gate_bias is not None:
            perceiver_dict["gate_bias"] = float(getattr(gate_bias, "data", gate_bias))

        bias = getattr(perceiver, "bias", None)
        if bias is not None:
            perceiver_dict["bias"] = float(getattr(bias, "data", bias))

        if perceiver_dict:
            data[name] = perceiver_dict

    if data:
        np.savez_compressed(weight_path(agent.name), **data)


def load_weights(agent: Any) -> None:
    path = weight_path(agent.name)
    if not path.exists():
        return

    loaded = np.load(path, allow_pickle=True)
    for name, block in agent.graph.blocks.items():
        if name not in loaded:
            continue

        perceiver = getattr(block, "perceiver", None)
        if perceiver is None:
            continue

        entry = loaded[name].item()

        _assign_weights(perceiver, "input_weights", entry)
        _assign_weights(perceiver, "memory_weights", entry)
        _assign_weights(perceiver, "gate_weights", entry)

        if "gate_bias" in entry and hasattr(perceiver, "gate_bias"):
            perceiver.gate_bias.data = float(entry["gate_bias"])

        if "bias" in entry and hasattr(perceiver, "bias"):
            perceiver.bias.data = float(entry["bias"])


def save_memory(agent: Any, limit: int = 100) -> None:
    """Guarda últimas experiencias del agente."""

    memory = getattr(agent.memory_system, "memory", None)
    if memory is None:
        return

    buffer = list(getattr(memory, "buffer", []))[-limit:]
    serializable = []
    for episode in buffer:
        serializable.append(
            {
                "input": _to_serializable(episode.get("input")),
                "target": _to_serializable(episode.get("target")),
                "output": _to_serializable(episode.get("output")),
                "loss": float(episode.get("loss", 0.0)),
                "attention": _to_serializable(episode.get("attention", {})),
            }
        )

    with memory_path(agent.name).open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh)


def load_memory(agent: Any) -> None:
    path = memory_path(agent.name)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as fh:
        episodes = json.load(fh)

    memory = getattr(agent.memory_system, "memory", None)
    if memory is None:
        return

    for episode in episodes:
        memory.store(
            _from_serializable(episode.get("input")),
            _from_serializable(episode.get("target")),
            _from_serializable(episode.get("output")),
            float(episode.get("loss", 0.0)),
            _from_serializable(episode.get("attention", {})),
        )


def _assign_weights(perceiver: Any, attr_name: str, entry: Dict[str, Any]) -> None:
    if attr_name not in entry or not hasattr(perceiver, attr_name):
        return

    stored_values: Iterable[float] = entry[attr_name]
    target_weights = getattr(perceiver, attr_name)
    for weight_obj, saved in zip(target_weights, stored_values):
        weight_obj.data = float(saved)


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "data"):
        return float(value.data)
    return value


def _from_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _from_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    return value
