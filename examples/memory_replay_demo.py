from __future__ import annotations

import os
import threading
import sys
from typing import Any

import numpy as np

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.trm_act_block import TRM_ACT_Block
from core.training.trainer import GraphTrainer
from core.monitor.global_state import set_graph

def main() -> None:
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("reasoner", TRM_ACT_Block(n_in=2, n_hidden=6, max_steps=3))
    graph.add_block("decision", CognitiveBlock(n_inputs=6, n_hidden=3, n_outputs=1))
    graph.connect("input", "reasoner")
    graph.connect("reasoner", "decision")

    trainer = GraphTrainer(graph, lr=0.01)
    memory_system = graph.memory_system
    _maybe_launch_dashboard(graph)

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


def _maybe_launch_dashboard(graph: Any) -> None:
    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError:
        return

    set_graph(graph)

    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError:
        return

    st.session_state.graph = graph

    def _run_streamlit() -> None:
        os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
        os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        from streamlit.web import cli as stcli  # type: ignore

        sys.argv = [
            "streamlit",
            "run",
            os.path.join(os.getcwd(), "dashboard", "app_dashboard.py"),
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
        stcli.main()

    if not any(t.name == "cognitive-dashboard" for t in threading.enumerate()):
        thread = threading.Thread(target=_run_streamlit, name="cognitive-dashboard", daemon=True)
        thread.start()


if __name__ == "__main__":
    main()
