from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block


def build_hybrid_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=1, n_hidden=4, max_steps=4))
    graph.add_block("decision", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.connect("sensor", "reasoner")
    graph.connect("reasoner", "decision")
    return graph


def test_forward_hybrid_graph() -> None:
    graph = build_hybrid_graph()
    outputs = graph.forward({"sensor": [0.3]})
    assert set(outputs.keys()) == {"sensor", "reasoner", "decision"}
    for tensor in outputs.values():
        arr = tensor.data
        assert isinstance(arr, np.ndarray)
        assert np.all(np.isfinite(arr))


def test_reset_states_hybrid() -> None:
    graph = build_hybrid_graph()
    graph.forward({"sensor": [0.5]})
    graph.reset_states()
    # TRM_ACT_Block debe resetear z a ceros
    reasoner = graph.blocks["reasoner"]
    assert isinstance(reasoner, TRM_ACT_Block)
    assert np.allclose(reasoner.z.data, 0.0)


def test_small_variation_stability() -> None:
    graph = build_hybrid_graph()
    out1 = graph.forward({"sensor": [0.2]})
    out2 = graph.forward({"sensor": [0.21]})
    val1 = float(out1["decision"].data.reshape(-1)[0])
    val2 = float(out2["decision"].data.reshape(-1)[0])
    assert abs(val1 - val2) < 0.3
