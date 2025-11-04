"""Demostración simple del sistema evolutivo cognitivo."""

from __future__ import annotations

import numpy as np

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.evolution.evolution_manager import EvolutionManager
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def make_base_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=2, n_hidden=6, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))

    graph.connect("input", "reasoner")
    graph.connect("reasoner", "decision")
    return graph


def main() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    base_graph = make_base_graph()
    manager = EvolutionManager(base_graph, GraphTrainer, X, Y, pop_size=5)
    manager.evolve(generations=4)


if __name__ == "__main__":
    main()
