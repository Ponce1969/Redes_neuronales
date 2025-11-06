"""Interactive Streamlit visualiser for the cognitive graph using PyG."""

from __future__ import annotations

import sys
from typing import Any

import streamlit as st

try:  # Soft dependency guard so we can show a friendly message in Streamlit.
    import torch
    from torch_geometric.utils import to_networkx
except ModuleNotFoundError:  # pragma: no cover - executed within Streamlit runtime
    st.set_page_config(page_title="🧠 Cognitive Graph Interactive Visualizer", layout="wide")
    st.error(
        "Este dashboard requiere PyTorch y torch-geometric instalados. "
        "Ejecuta `uv pip install torch torch-geometric` y vuelve a intentarlo."
    )
    st.stop()

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.latent_planner_block import LatentPlannerBlock
from core.pyg_bridge import CognitiveGraphAdapter, GATReasoner


def build_demo_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(2, 4, 1))
    graph.add_block("planner", LatentPlannerBlock(2, 8))
    graph.add_block("decision", CognitiveBlock(8, 4, 1))
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")

    graph.forward({"sensor": [0.5, -0.2]})
    return graph


def prepare_graph_data(graph: CognitiveGraphHybrid) -> tuple[Any, nx.Graph, np.ndarray, np.ndarray, np.ndarray]:
    adapter = CognitiveGraphAdapter(graph)
    data = adapter.to_pyg()

    model = GATReasoner(in_dim=data.num_features, hidden_dim=8, heads=2, out_dim=1)
    with torch.no_grad():
        gat_output = model(data).detach().cpu().numpy().reshape(-1)

    nx_graph = to_networkx(data, to_undirected=True)
    activations = data.x[:, 0].detach().cpu().numpy().reshape(-1)
    plan_means = data.x[:, 1].detach().cpu().numpy().reshape(-1)
    return data, nx_graph, activations, plan_means, gat_output


def build_plotly_figure(
    data: Any,
    nx_graph: nx.Graph,
    plan_means: np.ndarray,
) -> go.Figure:
    pos = nx.spring_layout(nx_graph, seed=42)
    x_nodes = [pos[i][0] for i in range(nx_graph.number_of_nodes())]
    y_nodes = [pos[i][1] for i in range(nx_graph.number_of_nodes())]

    x_edges: list[float | None] = []
    y_edges: list[float | None] = []
    for src, dest in nx_graph.edges():
        x_edges += [pos[src][0], pos[dest][0], None]
        y_edges += [pos[src][1], pos[dest][1], None]

    edge_trace = go.Scatter(
        x=x_edges,
        y=y_edges,
        line=dict(width=2, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=list(data.node_names),
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=plan_means,
            size=60,
            colorbar=dict(title="Z_plan", thickness=12),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Red Cognitiva – Estados actuales", font=dict(size=18)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def render_details_table(
    data: Any,
    activations: np.ndarray,
    plan_means: np.ndarray,
    gat_output: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nodo": list(data.node_names),
            "Activación_media": activations,
            "Z_plan": plan_means,
            "Salida_GAT": gat_output,
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="🧠 Cognitive Graph Interactive Visualizer",
        layout="wide",
    )
    st.title("🧬 Cognitive Graph Interactive Visualizer")
    st.write("Interactúa con los nodos cognitivos y explora sus estados en tiempo real.")

    graph = build_demo_graph()
    data, nx_graph, activations, plan_means, gat_output = prepare_graph_data(graph)

    fig = build_plotly_figure(data, nx_graph, plan_means)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.subheader("📊 Datos de nodos")
    df = render_details_table(data, activations, plan_means, gat_output)
    st.dataframe(df, use_container_width=True)

    selected_node = st.selectbox("🔍 Seleccionar nodo", options=list(data.node_names))
    if selected_node:
        idx = list(data.node_names).index(selected_node)
        st.markdown(f"### 🧠 Nodo: `{selected_node}`")
        st.metric("Z_plan", f"{plan_means[idx]:.4f}")
        st.metric("Salida GAT", f"{gat_output[idx]:.4f}")
        st.info(
            "Integra esta vista con tus nodos remotos exponiendo `/api/graph/state` y reemplazando el loader "
            "demo por datos en vivo."
        )

    st.caption(
        "💡 Los colores reflejan la intensidad del pensamiento latente (z_plan). Refresca para actualizar valores "
        "o conecta un flujo en tiempo real con WebSockets."
    )


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()
