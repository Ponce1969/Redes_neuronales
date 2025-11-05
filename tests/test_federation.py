from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api.server import app
from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.federation.federated_client import FederatedClient
from core.federation.utils import average_weights, apply_weights, serialize_agent
from core.security import API_KEY_ENV
from core.society.agent import CognitiveAgent
from core.training.trainer import GraphTrainer


class DummyTrainer(GraphTrainer):  # pragma: no cover - same behaviour
    pass


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch: Any) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")


@pytest.fixture()
def agent(monkeypatch: Any, tmp_path: Path) -> CognitiveAgent:
    monkeypatch.chdir(tmp_path)
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(2, 2, 2))
    graph.add_block("decision", CognitiveBlock(2, 2, 1))
    graph.connect("input", "decision")
    agent = CognitiveAgent("AgentFed", graph, DummyTrainer)
    # Inicializar valores específicos para verificarlos luego
    for idx, weight in enumerate(agent.graph.blocks["input"].perceiver.input_weights):
        weight.data = float(idx + 1)
    return agent


def test_serialize_and_apply(agent: CognitiveAgent) -> None:
    payload = serialize_agent(agent)
    assert "input" in payload
    assert payload["input"]["perceiver_input"] == [1.0, 2.0]

    # modificar y aplicar nuevamente
    modified = payload.copy()
    modified["input"]["perceiver_input"] = [10.0, 20.0]
    apply_weights(agent, modified)
    values = [w.data for w in agent.graph.blocks["input"].perceiver.input_weights]
    assert values == [10.0, 20.0]


def test_average_weights() -> None:
    entries = [
        ({"block": {"param": [1.0, 3.0]}}, 1.0),
        ({"block": {"param": [3.0, 5.0]}}, 1.0),
    ]
    averaged = average_weights(entries)
    assert averaged["block"]["param"] == [2.0, 4.0]


def test_federated_server_flow(agent: CognitiveAgent, monkeypatch: Any) -> None:
    client = TestClient(app)

    # subir pesos
    response = client.post(
        "/federate/upload",
        json={"agent_name": agent.name, "weights": serialize_agent(agent)},
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 202

    # descargar promedio
    response = client.get("/federate/global", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    data = response.json()
    assert "weights" in data


def test_federated_client_roundtrip(agent: CognitiveAgent, monkeypatch: Any) -> None:
    client = TestClient(app)
    prefix = str(client.base_url)

    class DummySession:
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
            response = client.post(url.replace(prefix, ""), json=json, headers=headers)
            response.raise_for_status()
            return response

        def get(self, url: str, headers: dict[str, str], timeout: float) -> Any:
            response = client.get(url.replace(prefix, ""), headers=headers)
            response.raise_for_status()
            return response

    federated = FederatedClient(agent, str(client.base_url), api_key="test-key")
    monkeypatch.setattr(federated, "session", DummySession())

    upload_resp = federated.send_weights()
    assert upload_resp["status"] == "received"

    global_resp = federated.receive_global_weights()
    assert "weights" in global_resp
