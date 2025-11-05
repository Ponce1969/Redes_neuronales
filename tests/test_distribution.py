from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import base64

import numpy as np
import pytest

pytest.importorskip("fastapi")

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.distribution.distributor import CognitiveDistributor
from core.distribution.receiver import apply_shared_payload
from core.persistence.file_manager import memory_path, weight_path
from core.persistence.serializer import save_memory, save_weights
from core.society.agent import CognitiveAgent
from core.training.trainer import GraphTrainer


class DummyTrainer(GraphTrainer):  # pragma: no cover - reuse base behaviour
    pass


def build_agent(name: str) -> CognitiveAgent:
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(2, 2, 2))
    graph.add_block("decision", CognitiveBlock(2, 2, 1))
    graph.connect("input", "decision")
    return CognitiveAgent(name, graph, DummyTrainer)


@pytest.fixture()
def agent(tmp_path: Path) -> CognitiveAgent:
    agent = build_agent("AgentTest")
    agent.memory_system.memory.store(
        np.array([0.0, 1.0]),
        np.array([1.0]),
        np.array([0.0]),
        0.1,
        {"focus": [0.5]},
    )
    return agent


def test_distributor_creates_payload(agent: CognitiveAgent, monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    responses = []

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200

        def raise_for_status(self) -> None:  # pragma: no cover - success path
            return

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        responses.append({"url": url, "headers": headers, "json": json})
        return DummyResponse()

    monkeypatch.setattr("core.security.get_api_headers", lambda api_key=None: {"x-api-key": api_key or "secret"})

    dist = CognitiveDistributor("http://localhost:8000", "secret", timeout=0.1)
    monkeypatch.setattr(dist.session, "post", fake_post)

    save_weights(agent)
    save_memory(agent)

    assert dist.share_agent(agent) is True
    assert responses
    payload = responses[0]["json"]
    assert payload["agent_name"] == "AgentTest"
    assert payload["memory"]
    assert payload["weights"]


def test_apply_shared_payload(agent: CognitiveAgent, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)

    save_weights(agent)
    save_memory(agent)

    weights_data = weight_path(agent.name).read_bytes()
    encoded = base64.b64encode(weights_data).decode("utf-8")

    payload = {
        "memory": [
            {
                "input": [1.0, 0.0],
                "target": [0.0],
                "output": [0.9],
                "loss": 0.05,
                "attention": {"focus": [0.7]},
            }
        ],
        "weights": encoded,
    }

    result = apply_shared_payload(agent, payload)
    assert result["received"] == 1
    assert result["weights_applied"] is True
