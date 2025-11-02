from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def main() -> None:
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=2, n_hidden=6, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))

    graph.connect("input", "reasoner")
    graph.connect("reasoner", "decision")

    trainer = GraphTrainer(graph, lr=0.01)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    for epoch in range(100):
        batch_inputs = [{"input": X[i]} for i in range(len(X))]
        batch_targets = [Y[i] for i in range(len(Y))]
        trainer.train_step(batch_inputs, batch_targets)

        if epoch % 25 == 0:
            graph.monitor.logger.log("INFO", f"Epoch {epoch} completado")

    print("\n--- RESUMEN FINAL ---")
    graph.monitor.summary()


if __name__ == "__main__":
    main()
