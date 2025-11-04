"""Dependencias compartidas para el servidor cognitivo."""

from __future__ import annotations

import threading
from typing import Dict, List

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.persistence import PersistenceManager
from core.society.agent import CognitiveAgent
from core.society.society_manager import SocietyManager
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer


def build_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=2, n_hidden=6, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))

    graph.connect("input", "reasoner")
    graph.connect("reasoner", "decision")
    return graph


def build_society(n_agents: int = 3) -> SocietyManager:
    agents = [CognitiveAgent(f"Agent_{idx}", build_graph(), GraphTrainer) for idx in range(n_agents)]
    return SocietyManager(agents)


class CognitiveAppState:
    """Estado compartido del servidor cognitivo."""

    def __init__(self, society: SocietyManager) -> None:
        self.society = society
        self.lock = threading.RLock()
        self.predict_calls = 0
        self.feedback_calls = 0
        self.metadata: Dict[str, List[float]] = {"loss_history": []}

    def record_predict(self) -> None:
        with self.lock:
            self.predict_calls += 1

    def record_feedback(self, loss: float) -> None:
        with self.lock:
            self.feedback_calls += 1
            self.metadata.setdefault("loss_history", []).append(float(loss))


society = build_society()
app_state = CognitiveAppState(society)
persistence_manager = PersistenceManager(society)
persistence_manager.load_all()


def get_app_state() -> CognitiveAppState:
    return app_state


def get_persistence_manager() -> PersistenceManager:
    return persistence_manager
