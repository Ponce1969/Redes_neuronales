from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch es requerido para el puente PyG")
pytest.importorskip("torch_geometric", reason="PyTorch Geometric es requerido para el puente PyG")
from torch import nn
from torch_geometric.data import Data

from core.pyg_bridge import CognitiveGraphAdapter, GraphTrainer


class DummyBlock:
    def __init__(self, activation: float, plan: np.ndarray | None = None) -> None:
        self.last_activation = activation
        self._plan = plan

    def get_last_plan(self) -> np.ndarray | None:
        return self._plan


class DummyGraph:
    def __init__(self) -> None:
        self.blocks = {
            "sensor": DummyBlock(0.5, plan=np.array([0.1, 0.9], dtype=np.float32)),
            "decision": DummyBlock(0.3),
        }
        self.connections = {"sensor": [], "decision": ["sensor"]}


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, data: Data) -> torch.Tensor:
        return self.linear(data.x)


@pytest.fixture()
def pyg_data() -> Data:
    graph = DummyGraph()
    adapter = CognitiveGraphAdapter(graph)
    return adapter.to_pyg()


def test_adapter_outputs_pyg_data(pyg_data: Data) -> None:
    assert pyg_data.x.shape == (2, 2)
    assert isinstance(pyg_data.edge_index, torch.Tensor)
    assert pyg_data.edge_index.shape[0] == 2
    assert hasattr(pyg_data, "node_names")
    assert list(pyg_data.node_names) == ["sensor", "decision"]


def test_graph_trainer_step_and_infer(pyg_data: Data) -> None:
    model = DummyModel()
    trainer = GraphTrainer(model, lr=0.01)
    target = torch.zeros(pyg_data.num_nodes)

    loss = trainer.train_step(pyg_data, target)
    assert isinstance(loss, float)

    predictions = trainer.infer(pyg_data)
    assert predictions.shape[0] == pyg_data.num_nodes
