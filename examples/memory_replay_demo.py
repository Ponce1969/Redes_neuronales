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
    memory_system = graph.memory_system

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    for epoch in range(300):
        inputs = [{"input": X[i]} for i in range(4)]
        targets = [Y[i] for i in range(4)]
        loss = trainer.train_step(inputs, targets)

        if epoch % 100 == 0:
            graph.monitor.logger.log("INFO", f"Epoch {epoch}, pérdida={loss:.4f}")

    print("\n🌙 Iniciando fase de sueño cognitivo...")
    for _ in range(3):
        memory_system.sleep_and_replay()

    print("\n--- RESUMEN DE MEMORIA ---")
    print(f"Episodios almacenados: {len(memory_system.memory)}")


if __name__ == "__main__":
    main()
