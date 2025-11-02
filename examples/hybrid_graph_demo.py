from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block


def main() -> None:
    graph = CognitiveGraphHybrid()

    graph.add_block("sensor", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=1, n_hidden=8, max_steps=5))
    graph.add_block("decision", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))

    graph.connect("sensor", "reasoner")
    graph.connect("reasoner", "decision")

    graph.summary()

    print("\n--- Ejecución de razonamiento mixto ---")
    sequence = [0.2, 0.5, 0.9, 0.4]
    for step, value in enumerate(sequence):
        out = graph.forward({"sensor": [value]})
        print(f"Step {step} | Input={value:.2f}")
        for name, tensor in out.items():
            val = float(tensor.data.reshape(-1)[0])
            print(f"   {name:<10s} -> {val:.4f}")


if __name__ == "__main__":
    main()
