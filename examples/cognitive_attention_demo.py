from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block


def main() -> None:
    graph = CognitiveGraphHybrid()

    graph.add_block("vision", CognitiveBlock(n_inputs=4, n_hidden=4, n_outputs=2))
    graph.add_block("memory", TRM_ACT_Block(n_in=2, n_hidden=6, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))

    graph.connect("vision", "memory")
    graph.connect("memory", "decision")

    print("=== Atención cognitiva dinámica ===")
    for step in range(4):
        inputs = {"vision": np.random.rand(4)}
        outputs = graph.forward(inputs)
        print(f"\nStep {step}")
        for name, tensor in outputs.items():
            value = float(tensor.data.reshape(-1)[0])
            print(f"  {name:<10s} -> {value:.4f}")
        if graph.last_attention.get("decision"):
            print("  attention ->", {k: float(v[0]) for k, v in graph.last_attention["decision"].items()})


if __name__ == "__main__":
    main()
