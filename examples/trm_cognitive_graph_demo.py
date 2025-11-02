from __future__ import annotations

import numpy as np

from core.cognitive_graph_trm import CognitiveGraphTRM
from core.trm_act_block import TRM_ACT_Block


def main() -> None:
    graph = CognitiveGraphTRM()
    graph.add_block("perception", TRM_ACT_Block(n_in=1, n_hidden=8, max_steps=6))
    graph.add_block("reason", TRM_ACT_Block(n_in=1, n_hidden=10, max_steps=6))
    graph.add_block("decision", TRM_ACT_Block(n_in=1, n_hidden=6, max_steps=6))

    graph.connect("perception", "reason")
    graph.connect("reason", "decision")

    graph.summary()

    inputs = [0.1, 0.7, 0.2, 0.9, 0.4]
    print("\n--- TRM CognitiveGraph Execution ---")
    for t, x in enumerate(inputs):
        out = graph.step_numeric({"perception": [x]})
        print(f"Step {t} | input={x:.2f} -> outputs:")
        for name, value in out.items():
            val = float(value.reshape(-1)[0])
            print(f"   {name}: {val:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
