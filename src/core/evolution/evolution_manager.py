"""Gestor evolutivo para población de grafos cognitivos."""

from __future__ import annotations

import copy
from typing import Sequence, Type

import numpy as np

try:
    from src.core.evolution.crossover import crossover_weights, crossover_scalar  # type: ignore
    from src.autograd.value import Value  # type: ignore
except ModuleNotFoundError:
    from core.evolution.crossover import crossover_weights, crossover_scalar  # type: ignore
    from autograd.value import Value  # type: ignore

from core.evolution.population import CognitivePopulation


class EvolutionManager:
    """Coordina evaluaciones, selección y cruce entre grafos."""

    def __init__(
        self,
        base_graph: object,
        trainer_class: Type,
        data_X: Sequence[np.ndarray],
        data_Y: Sequence[np.ndarray],
        pop_size: int = 6,
    ) -> None:
        self.population = CognitivePopulation(base_graph, pop_size)
        self.trainer_class = trainer_class
        self.data_X = data_X
        self.data_Y = data_Y

    # ------------------------------------------------------------------
    # Ciclo evolutivo
    # ------------------------------------------------------------------
    def evolve(self, generations: int = 5) -> None:
        for gen in range(generations):
            fitness = self.population.evaluate(self.trainer_class, self.data_X, self.data_Y)
            best_idx = self.population.select_best(k=2)
            best_graphs = [self.population.graphs[i] for i in best_idx]
            print(f"Gen {gen}: best fitness={fitness[best_idx[0]]:.4f}")

            new_graphs = []
            for _ in range(len(self.population.graphs)):
                parent1, parent2 = np.random.choice(best_graphs, 2, replace=True)
                child = self._cross_graphs(parent1, parent2)
                new_graphs.append(child)

            self.population.graphs = new_graphs
            self.population.generation += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cross_graphs(self, g1: object, g2: object) -> object:
        child = copy.deepcopy(g1)

        for name in g1.blocks.keys():  # type: ignore[attr-defined]
            b1 = g1.blocks[name]
            b2 = g2.blocks[name]
            child_block = child.blocks[name]

            if hasattr(b1, "perceiver") and hasattr(b2, "perceiver"):
                self._cross_perceiver(child_block.perceiver, b1.perceiver, b2.perceiver)

            if hasattr(b1, "decision_weights") and hasattr(b2, "decision_weights"):
                for idx, (w1, w2) in enumerate(zip(b1.decision_weights, b2.decision_weights)):
                    if isinstance(w1, Value) and isinstance(w2, Value):
                        child_block.decision_weights[idx].data = crossover_scalar(w1.data, w2.data)
                    else:  # fallback para tensores tipo Tensor
                        child_block.decision_weights[idx].data = crossover_weights(w1.data, w2.data)

            if hasattr(b1, "decision_bias") and hasattr(b2, "decision_bias"):
                if isinstance(b1.decision_bias, Value) and isinstance(b2.decision_bias, Value):
                    child_block.decision_bias.data = crossover_scalar(b1.decision_bias.data, b2.decision_bias.data)
                else:
                    child_block.decision_bias.data = crossover_weights(
                        np.array([[b1.decision_bias.data]], dtype=np.float32),
                        np.array([[b2.decision_bias.data]], dtype=np.float32),
                    )[0, 0]

            if hasattr(b1, "W_in") and hasattr(b2, "W_in"):
                child_block.W_in.data = crossover_weights(b1.W_in.data, b2.W_in.data)
            if hasattr(b1, "W_out") and hasattr(b2, "W_out"):
                child_block.W_out.data = crossover_weights(b1.W_out.data, b2.W_out.data)
            if hasattr(b1, "W_z") and hasattr(b2, "W_z"):
                child_block.W_z.data = crossover_weights(b1.W_z.data, b2.W_z.data)
            if hasattr(b1, "W_halt") and hasattr(b2, "W_halt"):
                child_block.W_halt.data = crossover_weights(b1.W_halt.data, b2.W_halt.data)

        return child

    def _cross_perceiver(self, child_perceiver, p1, p2) -> None:
        if hasattr(p1, "input_weights") and hasattr(p2, "input_weights"):
            for idx, (w1, w2) in enumerate(zip(p1.input_weights, p2.input_weights)):
                if isinstance(w1, Value) and isinstance(w2, Value):
                    child_perceiver.input_weights[idx].data = crossover_scalar(w1.data, w2.data)
                else:
                    child_perceiver.input_weights[idx].data = crossover_weights(w1.data, w2.data)
        if hasattr(p1, "memory_weights") and hasattr(p2, "memory_weights"):
            for idx, (w1, w2) in enumerate(zip(p1.memory_weights, p2.memory_weights)):
                if isinstance(w1, Value) and isinstance(w2, Value):
                    child_perceiver.memory_weights[idx].data = crossover_scalar(w1.data, w2.data)
                else:
                    child_perceiver.memory_weights[idx].data = crossover_weights(w1.data, w2.data)
        if hasattr(p1, "gate_weights") and hasattr(p2, "gate_weights"):
            for idx, (w1, w2) in enumerate(zip(p1.gate_weights, p2.gate_weights)):
                if isinstance(w1, Value) and isinstance(w2, Value):
                    child_perceiver.gate_weights[idx].data = crossover_scalar(w1.data, w2.data)
                else:
                    child_perceiver.gate_weights[idx].data = crossover_weights(w1.data, w2.data)
        if hasattr(p1, "gate_bias") and hasattr(p2, "gate_bias"):
            if isinstance(p1.gate_bias, Value) and isinstance(p2.gate_bias, Value):
                child_perceiver.gate_bias.data = crossover_scalar(p1.gate_bias.data, p2.gate_bias.data)
            else:
                child_perceiver.gate_bias.data = crossover_weights(
                    np.array([[p1.gate_bias.data]], dtype=np.float32),
                    np.array([[p2.gate_bias.data]], dtype=np.float32),
                )[0, 0]
        if hasattr(p1, "bias") and hasattr(p2, "bias"):
            if isinstance(p1.bias, Value) and isinstance(p2.bias, Value):
                child_perceiver.bias.data = crossover_scalar(p1.bias.data, p2.bias.data)
            else:
                child_perceiver.bias.data = crossover_weights(
                    np.array([[p1.bias.data]], dtype=np.float32),
                    np.array([[p2.bias.data]], dtype=np.float32),
                )[0, 0]
