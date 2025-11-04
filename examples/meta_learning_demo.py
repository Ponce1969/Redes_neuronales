"""Demostración del meta-learning loop adaptativo."""

from __future__ import annotations

import numpy as np

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.meta.meta_controller import MetaLearningController
from core.memory.memory_replay import MemoryReplaySystem
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
    graph = build_graph()

    trainer = GraphTrainer(graph, lr=0.01)
    memory_system = MemoryReplaySystem(graph, graph.monitor)
    graph.memory_system = memory_system

    meta = MetaLearningController(trainer, memory_system, graph.monitor)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    epochs = 60
    for epoch in range(epochs):
        batch_inputs = [{"input": X[i]} for i in range(len(X))]
        batch_targets = [Y[i] for i in range(len(Y))]

        loss = trainer.train_step(batch_inputs, batch_targets)

        # Meta-control heurístico
        meta.observe_and_adjust(epoch, loss)
        meta.maybe_sleep(epoch)

    print("\n=== META LOOP FINAL ===")
    print(f"Learning rate final: {meta.lr:.5f}")
    print(f"Focus atencional final: {meta.focus:.3f}")
    print(f"Intervalo de sueño: {meta.sleep_interval}")


if __name__ == "__main__":
    main()
