"""Demostración de la Sociedad Cognitiva (Fase 22)."""

from __future__ import annotations

import numpy as np

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.society.agent import CognitiveAgent
from core.society.society_manager import SocietyManager
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def build_graph() -> CognitiveGraphHybrid:
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

    agents = [CognitiveAgent(f"Agent_{idx}", build_graph(), GraphTrainer) for idx in range(3)]
    society = SocietyManager(agents)

    society.run_cycle(X, Y, epochs=40, exchange_every=8, broadcast_top=True)

    best = society.top_agent()
    if best is not None:
        print(f"\n[Society] Mejor agente: {best.name} (performance={best.performance:.4f})")


if __name__ == "__main__":
    main()
