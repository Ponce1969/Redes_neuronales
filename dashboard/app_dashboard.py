"""Dashboard cognitivo basado en Streamlit.

Ejecutar con:
    streamlit run dashboard/app_dashboard.py
"""

from __future__ import annotations

import os
import sys

try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit es necesario para ejecutar el dashboard.") from exc

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.monitor.visualizer_streamlit import CognitiveVisualizer  # noqa: E402
from core.monitor.global_state import get_graph, load_state_snapshot  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Cognitive Dashboard", layout="wide")
    st.sidebar.title("Configuración del Dashboard")
    st.sidebar.markdown("Este panel visualiza el estado del CognitiveGraphHybrid.")

    graph = st.session_state.get("graph")
    if graph is None:
        graph = get_graph()

    if graph is None:
        snapshot = load_state_snapshot()
        if snapshot:
            st.info(
                "Mostrando datos desde el último snapshot persistido. Corre el entrenamiento para obtener datos en vivo."
            )
            graph = snapshot
            st.session_state["graph"] = graph
        else:
            st.warning(
                "⚠️ No hay instancia activa del grafo ni snapshot persistido. Ejecuta el entrenamiento para generar datos."
            )
            return
    else:
        st.session_state["graph"] = graph

    viz = CognitiveVisualizer(graph)
    viz.render_dashboard()


if __name__ == "__main__":
    main()
