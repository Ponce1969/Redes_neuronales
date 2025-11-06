"""Streamlit dashboard to visualise the cognitive graph via the PyG bridge."""

from __future__ import annotations

import sys
from typing import Any

import streamlit as st

try:  # Soft guard so the dashboard can show a friendly warning.
    import torch
    from torch_geometric.utils import to_networkx
except ModuleNotFoundError:  # pragma: no cover - executed in Streamlit runtime
    st.error(
        "Este dashboard requiere PyTorch y torch-geometric instalados. "
        "Por favor ejecuta `uv pip install torch torch-geometric`."
    )
    st.stop()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.latent_planner_block import LatentPlannerBlock
from core.pyg_bridge import CognitiveGraphAdapter, GCNReasoner


def build_demo_graph() -> CognitiveGraphHybrid:
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(2, 4, 1))
    graph.add_block("planner", LatentPlannerBlock(2, 8))
    graph.add_block("decision", CognitiveBlock(8, 4, 1))
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")

    graph.forward({"sensor": [0.5, -0.2]})
    return graph


def convert_to_networkx(graph: CognitiveGraphHybrid) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    adapter = CognitiveGraphAdapter(graph)
    data = adapter.to_pyg()

    model = GCNReasoner(in_dim=data.num_features, hidden_dim=8, out_dim=1)
    with torch.no_grad():
        gnn_output = model(data).detach().cpu()

    nx_graph = to_networkx(data, to_undirected=True)
    activations = data.x[:, 0].detach().cpu()
    plan_means = data.x[:, 1].detach().cpu()
    return data, nx_graph, activations, plan_means, gnn_output


def render_graph(data: Any, nx_graph: nx.Graph, activations: torch.Tensor, plan_means: torch.Tensor) -> None:
    plan_np = plan_means.numpy()
    colors = plt.cm.viridis((plan_np - plan_np.min()) / (np.ptp(plan_np) + 1e-5))
    pos = nx.spring_layout(nx_graph, seed=42)

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(nx_graph, pos, node_color=colors, node_size=800, ax=ax)
    nx.draw_networkx_edges(nx_graph, pos, width=2.0, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels={idx: name for idx, name in enumerate(data.node_names)},
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=plan_np.min(), vmax=plan_np.max()))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Intensidad z_plan")

    ax.set_axis_off()
    st.pyplot(fig)


def render_table(data: Any, activations: torch.Tensor, plan_means: torch.Tensor, gnn_output: torch.Tensor) -> None:
    df = pd.DataFrame(
        {
            "Node": list(data.node_names),
            "Activation_mean": activations.numpy().reshape(-1),
            "Z_plan_mean": plan_means.numpy().reshape(-1),
            "GNN_output": gnn_output.numpy().reshape(-1),
        }
    )
    st.dataframe(df, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="🧠 Cognitive Graph Visualization",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("🧩 Cognitive Graph Visualization – PyG Bridge")
    st.write("Visualización del grafo cognitivo con activaciones y planes latentes (`z_plan`).")

    graph = build_demo_graph()
    data, nx_graph, activations, plan_means, gnn_output = convert_to_networkx(graph)

    col_graph, col_table = st.columns([2, 1])
    with col_graph:
        st.subheader("🌐 Topología cognitiva")
        render_graph(data, nx_graph, activations, plan_means)
    with col_table:
        st.subheader("📊 Métricas de nodos")
        render_table(data, activations, plan_means, gnn_output)

    st.info(
        "💡 Refresca la página para regenerar la vista o adapta el loader para consumir el estado real "
        "desde tus nodos remotos."
    )


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()
