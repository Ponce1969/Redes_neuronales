"""Utilidades de serialización y promediado para aprendizaje federado."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from core.society.agent import CognitiveAgent


def average_weights(
    weight_entries: Iterable[
        dict[str, Mapping[str, Any]] | tuple[dict[str, Mapping[str, Any]], float]
    ]
) -> dict[str, dict[str, Any]]:
    """Promedia múltiples conjuntos de pesos (opcionalmente ponderados)."""

    prepared: list[tuple[dict[str, Mapping[str, Any]], float]] = []
    for entry in weight_entries:
        if isinstance(entry, tuple):
            weights, factor = entry
            prepared.append((weights, float(factor)))
        else:
            prepared.append((entry, 1.0))

    if not prepared:
        return {}

    accum: dict[tuple[str, str], np.ndarray] = {}
    weight_sum: dict[tuple[str, str], float] = {}

    for weights, factor in prepared:
        if factor <= 0:
            continue
        for block_name, params in weights.items():
            for param_name, value in params.items():
                arr = np.asarray(value, dtype=np.float32)
                key = (block_name, param_name)
                if key not in accum:
                    accum[key] = np.zeros_like(arr, dtype=np.float32)
                    weight_sum[key] = 0.0
                accum[key] += arr * factor
                weight_sum[key] += factor

    averaged: dict[str, dict[str, Any]] = {}
    for (block_name, param_name), total in accum.items():
        factor = weight_sum.get((block_name, param_name), 0.0)
        if factor <= 0:
            continue
        mean_value = total / factor
        serialized: Any
        if np.isscalar(mean_value) or mean_value.shape == ():
            serialized = float(np.asarray(mean_value, dtype=np.float32))
        else:
            serialized = np.asarray(mean_value, dtype=np.float32).tolist()
        block_data = averaged.setdefault(block_name, {})
        block_data[param_name] = serialized

    return averaged


def serialize_agent(agent: CognitiveAgent) -> dict[str, dict[str, Any]]:
    """Serializa en JSON-friendly los pesos relevantes del agente."""

    payload: dict[str, dict[str, Any]] = {}
    for name, block in agent.graph.blocks.items():
        block_data: dict[str, Any] = {}

        perceiver = getattr(block, "perceiver", None)
        if perceiver is not None:
            block_data["perceiver_input"] = _values_to_list(perceiver.input_weights)
            block_data["perceiver_memory"] = _values_to_list(perceiver.memory_weights)
            block_data["perceiver_gate"] = _values_to_list(perceiver.gate_weights)
            if getattr(perceiver, "gate_bias", None) is not None:
                block_data["perceiver_gate_bias"] = _value_to_float(perceiver.gate_bias)
            if getattr(perceiver, "bias", None) is not None:
                block_data["perceiver_bias"] = _value_to_float(perceiver.bias)

        reasoner = getattr(block, "reasoner", None)
        if reasoner is not None:
            block_data["reasoner_in"] = _values_to_list(reasoner.weights_in)
            block_data["reasoner_mem"] = _values_to_list(reasoner.weights_mem)
            if getattr(reasoner, "bias", None) is not None:
                block_data["reasoner_bias"] = _value_to_float(reasoner.bias)

        decision_weights = getattr(block, "decision_weights", None)
        if decision_weights is not None:
            block_data["decision_weights"] = _values_to_list(decision_weights)
        if getattr(block, "decision_bias", None) is not None:
            block_data["decision_bias"] = _value_to_float(block.decision_bias)

        if block_data:
            payload[name] = block_data

    return payload


def apply_weights(agent: CognitiveAgent, data: Mapping[str, Mapping[str, Any]]) -> None:
    """Aplica pesos serializados a los bloques correspondientes del agente."""

    for name, params in data.items():
        block = agent.graph.blocks.get(name)
        if block is None:
            continue

        perceiver = getattr(block, "perceiver", None)
        if perceiver is not None:
            if "perceiver_input" in params:
                _assign_list(perceiver.input_weights, params["perceiver_input"])
            if "perceiver_memory" in params:
                _assign_list(perceiver.memory_weights, params["perceiver_memory"])
            if "perceiver_gate" in params:
                _assign_list(perceiver.gate_weights, params["perceiver_gate"])
            if "perceiver_gate_bias" in params and getattr(perceiver, "gate_bias", None) is not None:
                _assign_scalar(perceiver.gate_bias, params["perceiver_gate_bias"])
            if "perceiver_bias" in params and getattr(perceiver, "bias", None) is not None:
                _assign_scalar(perceiver.bias, params["perceiver_bias"])

        reasoner = getattr(block, "reasoner", None)
        if reasoner is not None:
            if "reasoner_in" in params:
                _assign_list(reasoner.weights_in, params["reasoner_in"])
            if "reasoner_mem" in params:
                _assign_list(reasoner.weights_mem, params["reasoner_mem"])
            if "reasoner_bias" in params and getattr(reasoner, "bias", None) is not None:
                _assign_scalar(reasoner.bias, params["reasoner_bias"])

        decision_weights = getattr(block, "decision_weights", None)
        if decision_weights is not None and "decision_weights" in params:
            _assign_list(decision_weights, params["decision_weights"])
        if getattr(block, "decision_bias", None) is not None and "decision_bias" in params:
            _assign_scalar(block.decision_bias, params["decision_bias"])


def _values_to_list(values: Iterable[Any]) -> list[Any]:
    return [_value_to_float(v) for v in values]


def _value_to_float(value: Any) -> float:
    return float(getattr(value, "data", value))


def _assign_list(target: Iterable[Any], source: Iterable[Any]) -> None:
    for dest, src in zip(target, source):
        dest.data = float(getattr(src, "data", src))


def _assign_scalar(target: Any, value: Any) -> None:
    target.data = float(getattr(value, "data", value))
