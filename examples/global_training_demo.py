from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def build_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=2, n_hidden=8, max_steps=4))
    graph.add_block("decision", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.connect("sensor", "reasoner")
    graph.connect("reasoner", "decision")
    return graph


def main() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    graph = build_graph()
    trainer = GraphTrainer(graph, lr=0.01)
    trainer.summary()

    for epoch in range(800):
        batch_inputs = [{"sensor": X[i]} for i in range(len(X))]
        batch_targets = [Y[i] for i in range(len(Y))]
        loss = trainer.train_step(batch_inputs, batch_targets)
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss={loss:.6f}")

    print("\n--- Predicciones XOR ---")
    for x, y_true in zip(X, Y):
        outputs = graph.forward({"sensor": x})
        y_pred = float(list(outputs.values())[-1].data.reshape(-1)[0])
        print(f"{x} -> {y_pred:.3f} (target={y_true[0]:.1f})")


if __name__ == "__main__":
    main()
