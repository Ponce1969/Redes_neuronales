from __future__ import annotations

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def build_simple_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=1, n_hidden=4, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1))
    graph.connect("sensor", "reasoner")
    graph.connect("reasoner", "decision")
    return graph


def test_graph_trainer_collects_params() -> None:
    graph = build_simple_graph()
    trainer = GraphTrainer(graph, lr=0.01)
    assert len(trainer.params) > 0


def test_graph_trainer_train_step_runs() -> None:
    graph = build_simple_graph()
    trainer = GraphTrainer(graph, lr=0.01)
    inputs = [{"sensor": np.array([0.5], dtype=np.float32)}]
    targets = [np.array([0.7], dtype=np.float32)]
    loss = trainer.train_step(inputs, targets)
    assert isinstance(loss, float)
    assert loss >= 0.0
