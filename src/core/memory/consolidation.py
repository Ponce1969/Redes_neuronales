from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    try:
        from src.core.training.trainer import GraphTrainer  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        from core.training.trainer import GraphTrainer  # type: ignore


class MemoryConsolidation:
    """Reproduce experiencias almacenadas para consolidar el aprendizaje."""

    def __init__(self, graph: Any, episodic_memory: Any, replay_factor: int = 2) -> None:
        self.graph = graph
        self.memory = episodic_memory
        self.replay_factor = replay_factor
        # Importación diferida para evitar ciclos
        if TYPE_CHECKING:  # pragma: no cover
            trainer_cls = GraphTrainer  # type: ignore[name-defined]
        else:
            try:
                from src.core.training.trainer import GraphTrainer as trainer_cls  # type: ignore
            except ModuleNotFoundError:
                from core.training.trainer import GraphTrainer as trainer_cls  # type: ignore
        self.trainer = trainer_cls(graph, lr=0.005)

    def replay_phase(self) -> float:
        """Realiza una fase de replay con las mejores experiencias."""
        best_batch: List[dict[str, Any]] = self.memory.best_experiences(top_k=8)
        if not best_batch:
            return 0.0

        inputs = [self._clone_input(ep["input"]) for ep in best_batch]
        targets = [self._clone_target(ep["target"]) for ep in best_batch]

        total_loss = 0.0
        for _ in range(max(1, self.replay_factor)):
            loss = self.trainer.train_step(inputs, targets)
            total_loss += float(loss)
        return total_loss / max(1, self.replay_factor)

    def _clone_input(self, item: Any) -> Any:
        if isinstance(item, dict):
            return {k: self._clone_target(v) for k, v in item.items()}
        return self._clone_target(item)

    def _clone_target(self, item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.copy()
        if isinstance(item, (list, tuple)):
            return np.array(item, dtype=np.float32)
        return np.array([item], dtype=np.float32)
