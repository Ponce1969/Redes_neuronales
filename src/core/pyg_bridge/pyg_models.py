"""Graph neural network blocks to reason over cognitive graphs."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv, GCNConv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch y torch_geometric son requeridos para usar los modelos PyG."
    ) from exc


class _BaseReasoner(nn.Module):
    """Base helper with generic init to keep typing sane."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Any) -> torch.Tensor:  # pragma: no cover - interface placeholder
        raise NotImplementedError


class GCNReasoner(_BaseReasoner):
    """Two-layer GCN to propagate activations across the cognitive graph."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 16, out_dim: int = 1) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GATReasoner(_BaseReasoner):
    """Graph attention network with two stages for cognitive graphs."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 8, heads: int = 2, out_dim: int = 1) -> None:
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.0)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.0)
        self.act = nn.ELU()

    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.act(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x
