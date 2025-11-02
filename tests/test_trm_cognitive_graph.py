from __future__ import annotations

import numpy as np

from core.cognitive_graph_trm import CognitiveGraphTRM
from core.trm_act_block import TRM_ACT_Block


def build_graph_small() -> CognitiveGraphTRM:
    graph = CognitiveGraphTRM()
    graph.add_block("p", TRM_ACT_Block(n_in=1, n_hidden=4, max_steps=4))
    graph.add_block("r", TRM_ACT_Block(n_in=1, n_hidden=5, max_steps=4))
    graph.add_block("d", TRM_ACT_Block(n_in=1, n_hidden=3, max_steps=4))
    graph.connect("p", "r")
    graph.connect("r", "d")
    return graph


def test_forward_runs() -> None:
    graph = build_graph_small()
    out = graph.step_numeric({"p": [0.5]})
    assert {"p", "r", "d"}.issubset(out.keys())
    for value in out.values():
        assert isinstance(value, np.ndarray)
        assert np.all(np.isfinite(value))


def test_reset_states_no_error() -> None:
    graph = build_graph_small()
    graph.step_numeric({"p": [0.2]})
    graph.reset_states()
    for block in graph.blocks.values():
        assert hasattr(block, "z")
        assert np.all(block.z.data == 0)


def test_stability_small_variation() -> None:
    graph = build_graph_small()
    out1 = graph.step_numeric({"p": [0.30]})
    out2 = graph.step_numeric({"p": [0.32]})
    diff = abs(float(out1["d"].reshape(-1)[0]) - float(out2["d"].reshape(-1)[0]))
    assert diff < 0.25, f"Inestable: diff={diff}"
