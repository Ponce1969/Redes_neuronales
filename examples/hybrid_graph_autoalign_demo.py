from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block


def main() -> None:
    graph = CognitiveGraphHybrid()

    graph.add_block("sensor", CognitiveBlock(n_inputs=1, n_hidden=3, n_outputs=2))
    graph.add_block("memory", TRM_ACT_Block(n_in=2, n_hidden=8, max_steps=5))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=8, n_hidden=6, max_steps=4))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))

    graph.connect("sensor", "memory")
    graph.connect("memory", "reasoner")
    graph.connect("reasoner", "decision")

    graph.summary()

    print("\n--- Ejecución cognitiva con AutoAlign ---")
    for idx, value in enumerate([0.1, 0.6, 0.9, 0.3]):
        outputs = graph.forward({"sensor": [value]})
        print(f"Step {idx} | input={value:.2f}")
        for name, tensor in outputs.items():
            val = float(tensor.data.reshape(-1)[0])
            print(f"  {name:<10s} -> {val:.4f}")


if __name__ == "__main__":
    main()
