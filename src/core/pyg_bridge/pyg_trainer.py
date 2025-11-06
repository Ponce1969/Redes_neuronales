"""Utilities to train PyTorch Geometric models on cognitive graphs."""

from __future__ import annotations

from typing import Callable, Optional, Type

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
    from torch.optim import Adam, Optimizer
    from torch_geometric.data import Data
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch y torch_geometric son requeridos para usar el entrenador PyG."
    ) from exc


class GraphTrainer:
    """Entrena modelos PyG sobre grafos cognitivos adaptados."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        optimizer_cls: Type[Optimizer] = Adam,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.criterion = loss_fn or nn.MSELoss()

    def train_step(self, data: Data, target_values: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)
        target = target_values.to(self.device).float().view(-1)
        out = self.model(data).view(-1)
        loss = self.criterion(out, target)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def infer(self, data: Data) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(data.to(self.device)).cpu()
