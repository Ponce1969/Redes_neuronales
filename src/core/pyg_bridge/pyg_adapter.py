"""Adapter to convert CognitiveGraphHybrid instances into PyG Data objects."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - handled later
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - handled later
    Data = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as torch_types
    from torch_geometric.data import Data as DataType


class CognitiveGraphAdapter:
    """Converts a cognitive graph into a PyG ``Data`` object."""

    def __init__(self, cognitive_graph: Any) -> None:
        self._require_dependencies()
        self.graph = cognitive_graph

    @staticmethod
    def _require_dependencies() -> None:
        if torch is None or Data is None:
            raise ImportError(
                "PyTorch y torch_geometric son requeridos para usar el puente PyG."
            )

    def _collect_node_features(self, node_names: List[str]) -> "torch.Tensor":
        features: List[List[float]] = []
        for name in node_names:
            block = self.graph.blocks[name]
            activation = float(getattr(block, "last_activation", 0.0))
            plan_mean = 0.0
            get_plan = getattr(block, "get_last_plan", None)
            if callable(get_plan):
                plan = get_plan()
                if plan is not None:
                    plan_mean = float(np.asarray(plan, dtype=np.float32).mean())
            features.append([activation, plan_mean])
        return torch.tensor(features, dtype=torch.float32)  # type: ignore[call-arg]

    def _collect_edges(self, node_index: Dict[str, int]) -> "torch.Tensor":
        edges: List[List[int]] = []
        for dest, sources in self.graph.connections.items():
            if dest not in node_index:
                continue
            dest_idx = node_index[dest]
            for src in sources:
                if src not in node_index:
                    continue
                edges.append([node_index[src], dest_idx])
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()  # type: ignore[call-arg]
        return torch.empty((2, 0), dtype=torch.long)  # type: ignore[call-arg]

    def to_pyg(self) -> "Data":
        node_names = list(self.graph.blocks.keys())
        if not node_names:
            raise ValueError("El grafo cognitivo no contiene bloques registrados.")

        node_index = {name: idx for idx, name in enumerate(node_names)}
        x = self._collect_node_features(node_names)
        edge_index = self._collect_edges(node_index)

        data = Data(x=x, edge_index=edge_index)
        data.node_names = node_names  # type: ignore[attr-defined]
        return data
