"""Gestión de poblaciones de grafos cognitivos para evolución ligera."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np


class CognitivePopulation:
    """Mantiene múltiples instancias de un grafo cognitivo y su fitness."""

    def __init__(self, base_graph: object, size: int = 5) -> None:
        if size < 2:
            raise ValueError("La población debe tener al menos 2 individuos")

        self.graphs = [copy.deepcopy(base_graph) for _ in range(size)]
        self.fitness = np.zeros(size, dtype=np.float32)
        self.generation = 0

    # ------------------------------------------------------------------
    # Evaluación y selección
    # ------------------------------------------------------------------
    def evaluate(
        self,
        trainer_class,
        data_X: Sequence[np.ndarray],
        data_Y: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Evalúa cada grafo entrenándolo una pasada sobre los datos."""

        for idx, graph in enumerate(self.graphs):
            trainer = trainer_class(graph, lr=0.01)
            batch_inputs = [{"input": x} for x in data_X]
            batch_targets = [y for y in data_Y]
            loss = trainer.train_step(batch_inputs, batch_targets)
            self.fitness[idx] = -float(loss)

            reset = getattr(graph, "reset_states", None)
            if callable(reset):  # evitar que memoricen la sesión previa
                reset()

        return self.fitness.copy()

    def select_best(self, k: int = 2) -> np.ndarray:
        """Retorna los índices de los mejores individuos."""

        if k <= 0:
            raise ValueError("k debe ser positivo")

        k = min(k, len(self.graphs))
        return np.argsort(-self.fitness)[:k]
