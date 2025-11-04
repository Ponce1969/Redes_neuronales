"""Agentes cognitivos que envuelven un grafo y su entrenamiento."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np

from core.memory.memory_replay import MemoryReplaySystem


class CognitiveAgent:
    """Representa a un grafo cognitivo individual dentro de la sociedad."""

    def __init__(self, name: str, graph: Any, trainer_class: type) -> None:
        self.name = name
        self.graph = graph
        self.trainer_class = trainer_class

        self.memory_system = MemoryReplaySystem(graph, graph.monitor)
        self.trainer = trainer_class(graph, lr=0.01)
        self.performance = 0.0

    def train_once(self, X: Sequence[np.ndarray], Y: Sequence[np.ndarray]) -> float:
        """Ejecuta un paso de entrenamiento sobre el dataset dado."""

        batch_inputs = [{"input": x} for x in X]
        batch_targets = [y for y in Y]

        loss = self.trainer.train_step(batch_inputs, batch_targets)
        self.performance = -loss
        return float(loss)

    def receive_experiences(self, episodes: Iterable[dict[str, Any]]) -> None:
        """Recibe experiencias externas y las almacena en la memoria local."""

        for episode in episodes:
            self.memory_system.memory.store(
                deepcopy(episode.get("input")),
                deepcopy(episode.get("target")),
                deepcopy(episode.get("output")),
                float(episode.get("loss", 0.0)),
                deepcopy(episode.get("attention", {})),
            )

    def share_top_experiences(self, top_k: int = 4) -> list[dict[str, Any]]:
        """Devuelve una copia de las mejores experiencias actuales."""

        return [deepcopy(ep) for ep in self.memory_system.memory.best_experiences(top_k)]
