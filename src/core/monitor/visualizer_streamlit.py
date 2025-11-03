from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:  # Optional dependencies for dashboard rendering
    import streamlit as st  # type: ignore
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    _STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - dashboard optional
    st = None  # type: ignore
    pd = None  # type: ignore
    px = None  # type: ignore
    _STREAMLIT_AVAILABLE = False

try:
    from src.core.monitor.global_state import load_state_snapshot  # type: ignore
except ModuleNotFoundError:
    from core.monitor.global_state import load_state_snapshot  # type: ignore


class CognitiveVisualizer:
    """Visualizador Streamlit para el sistema cognitivo."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.monitor = getattr(graph, "monitor", None)
        self.memory_system = getattr(graph, "memory_system", None)
        self.snapshot = load_state_snapshot()
        self.monitor_snapshot: Dict[str, Any] | None = self.snapshot.get("monitor")
        self.memory_snapshot: list[Dict[str, Any]] | None = self.snapshot.get("memory")

    def render_dashboard(self) -> None:
        if not _STREAMLIT_AVAILABLE:  # pragma: no cover - feedback temprano
            raise RuntimeError(
                "Streamlit, pandas y plotly son necesarios para el dashboard. "
                "Instala las dependencias opcionales para utilizarlo."
            )

        if self.monitor is None and not self.monitor_snapshot:
            st.warning("No hay monitor cognitivo activo ni snapshot disponible. Ejecuta primero el entrenamiento.")
            return

        st.set_page_config(page_title="Cognitive Dashboard", layout="wide")
        st.title("🧠 Cognitive Dashboard – Sistema Cognitivo Híbrido")

        tabs = st.tabs(["📉 Pérdidas", "⚡ Activaciones", "🎯 Atención", "💭 Memoria episódica"])

        with tabs[0]:
            self._render_losses()
        with tabs[1]:
            self._render_activations()
        with tabs[2]:
            self._render_attention()
        with tabs[3]:
            self._render_memory()

        st.caption("Actualizado periódicamente. Usa Streamlit `rerun` para refrescar.")

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------
    def _render_losses(self) -> None:
        st.header("Pérdida Global")
        if self.monitor is not None:
            history = list(getattr(self.monitor, "loss_history", []))
        else:
            history = list(self.monitor_snapshot.get("loss_history", [])) if self.monitor_snapshot else []
        if history:
            df = pd.DataFrame({"Iteración": np.arange(len(history)), "Loss": history})
            fig = px.line(df, x="Iteración", y="Loss", title="Evolución de la pérdida global")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aún no se registraron pérdidas.")

    def _render_activations(self) -> None:
        st.header("Activaciones promedio por bloque")
        if self.monitor is not None:
            activations: Dict[str, np.ndarray] = getattr(self.monitor, "activations", {})
        else:
            raw = self.monitor_snapshot.get("activations", {}) if self.monitor_snapshot else {}
            activations = {name: np.asarray(values) for name, values in raw.items()}
        if activations:
            data = {
                "Bloque": list(activations.keys()),
                "Media Activación": [float(np.mean(v)) for v in activations.values()],
            }
            df = pd.DataFrame(data)
            fig = px.bar(df, x="Bloque", y="Media Activación", color="Bloque", title="Actividad promedio")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay activaciones registradas aún.")

    def _render_attention(self) -> None:
        st.header("Pesos de atención (última iteración)")
        if self.monitor is not None:
            attention: Dict[str, Dict[str, np.ndarray]] = getattr(self.monitor, "attention_weights", {})
        else:
            raw = self.monitor_snapshot.get("attention", {}) if self.monitor_snapshot else {}
            attention = {
                dest: {src: np.asarray(weights) for src, weights in sources.items()}
                for dest, sources in raw.items()
            }
        if attention:
            for dest, sources in attention.items():
                st.subheader(f"Destino: {dest}")
                df = pd.DataFrame({
                    "Fuente": list(sources.keys()),
                    "Peso Medio": [float(np.mean(w)) for w in sources.values()],
                })
                fig = px.bar(df, x="Fuente", y="Peso Medio", color="Fuente", title=f"Atención hacia {dest}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay pesos de atención disponibles.")

    def _render_memory(self) -> None:
        st.header("Memoria episódica (últimos episodios)")
        if self.memory_system is not None and hasattr(self.memory_system, "memory"):
            memory = self.memory_system.memory
            episodes = list(memory.buffer)[-10:]
        elif self.memory_snapshot:
            episodes = self.memory_snapshot[-10:]
        else:
            st.info("La memoria aún no contiene experiencias.")
            return

        data = [
            {
                "Input": str(ep["input"]),
                "Target": str(ep["target"]),
                "Loss": float(ep["loss"]),
                "Attention": str(ep.get("attention", {})),
            }
            for ep in episodes
        ]
        st.dataframe(pd.DataFrame(data))
