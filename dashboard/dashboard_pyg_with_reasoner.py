"""
Interactive PyG Visualizer con Reasoner Gates - Fase 32

Dashboard que combina visualización PyG del grafo cognitivo con gates del Reasoner,
mostrando qué bloques están activados en tiempo real.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

try:
    import torch
    from torch_geometric.utils import to_networkx
except ModuleNotFoundError:
    st.set_page_config(page_title="🧠 Cognitive Graph + Reasoner", layout="wide")
    st.error(
        "Este dashboard requiere PyTorch y torch-geometric. "
        "Ejecuta `uv pip install torch torch-geometric`"
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

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

API_URL = "http://localhost:8000"
REASONER_API = f"{API_URL}/reasoner"

st.set_page_config(
    page_title="🧠 Cognitive Graph + Reasoner",
    page_icon="🧠",
    layout="wide",
)

# ============================================================================
# FUNCIONES DE API
# ============================================================================


def get_reasoner_gates() -> Optional[Dict]:
    """Obtiene los últimos gates del Reasoner."""
    try:
        response = requests.get(f"{REASONER_API}/gates?n=1", timeout=2)
        response.raise_for_status()
        data = response.json()
        if data.get("gates_history"):
            return data["gates_history"][-1]
        return None
    except Exception:
        return None


def get_reasoner_status() -> Dict:
    """Obtiene el estado del Reasoner."""
    try:
        response = requests.get(f"{REASONER_API}/status", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# CONSTRUCCIÓN DEL GRAFO DEMO
# ============================================================================


def build_demo_graph() -> CognitiveGraphHybrid:
    """Construye grafo demo para visualización."""
    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(2, 4, 2))
    graph.add_block("planner", LatentPlannerBlock(2, 8, retain_plan=True))
    graph.add_block("memory", CognitiveBlock(8, 6, 4))
    graph.add_block("decision", CognitiveBlock(4, 3, 1))
    
    graph.connect("sensor", "planner")
    graph.connect("planner", "memory")
    graph.connect("memory", "decision")
    
    # Forward inicial para generar activaciones
    graph.forward({"sensor": [0.5, -0.2]})
    return graph


# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================


def prepare_graph_data(
    graph: CognitiveGraphHybrid, gates: Optional[Dict] = None
) -> tuple[Any, nx.Graph, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Prepara datos del grafo incluyendo gates del Reasoner."""
    adapter = CognitiveGraphAdapter(graph)
    data = adapter.to_pyg()
    
    # Modelo GAT para razonamiento
    model = GATReasoner(in_dim=data.num_features, hidden_dim=8, heads=2, out_dim=1)
    with torch.no_grad():
        gat_output = model(data).detach().cpu().numpy().reshape(-1)
    
    # Convertir a NetworkX
    nx_graph = to_networkx(data, to_undirected=True)
    
    # Extraer features
    activations = data.x[:, 0].detach().cpu().numpy().reshape(-1)
    plan_means = data.x[:, 1].detach().cpu().numpy().reshape(-1)
    
    # Convertir gates a array (si existen)
    gates_array = None
    if gates:
        gates_array = np.array([gates.get(i, 0.0) for i in range(len(graph.blocks))])
    
    return data, nx_graph, activations, plan_means, gat_output, gates_array


# ============================================================================
# VISUALIZACIÓN PLOTLY
# ============================================================================


def build_plotly_figure(
    nx_graph: nx.Graph,
    node_names: List[str],
    plan_means: np.ndarray,
    gates: Optional[np.ndarray] = None,
    color_by: str = "plan",
) -> go.Figure:
    """Construye figura interactiva con Plotly."""
    pos = nx.spring_layout(nx_graph, seed=42)
    x_nodes = [pos[i][0] for i in range(nx_graph.number_of_nodes())]
    y_nodes = [pos[i][1] for i in range(nx_graph.number_of_nodes())]
    
    # Edges
    x_edges: List[Optional[float]] = []
    y_edges: List[Optional[float]] = []
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
    
    # Determinar colores de nodos
    if color_by == "gates" and gates is not None:
        node_color = gates
        colorbar_title = "Gate Activation"
        colorscale = "Viridis"
    else:
        node_color = plan_means
        colorbar_title = "Plan Intensity"
        colorscale = "Inferno"
    
    # Crear texto de hover
    hover_text = []
    for i, name in enumerate(node_names):
        text = f"<b>{name}</b><br>"
        text += f"Plan: {plan_means[i]:.3f}<br>"
        if gates is not None:
            text += f"Gate: {gates[i]:.3f}<br>"
        hover_text.append(text)
    
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        marker=dict(
            size=30,
            color=node_color,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, thickness=15),
            line=dict(width=2, color="white"),
        ),
        text=node_names,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🧠 Cognitive Graph + Reasoner Visualization")
st.markdown("Visualización interactiva del grafo cognitivo con gates del Reasoner en tiempo real")

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Configuración")
    
    color_by = st.selectbox(
        "Colorear nodos por:",
        ["plan", "gates"],
        help="Plan: intensidad de plan latente | Gates: activación del Reasoner",
    )
    
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Intervalo (s)", 1, 10, 2, 1)
    
    st.markdown("---")
    st.subheader("📊 Estado del Reasoner")
    
    status_placeholder = st.empty()

# ============================================================================
# CONSTRUCCIÓN Y VISUALIZACIÓN
# ============================================================================

# Construir grafo demo
graph = build_demo_graph()
node_names = list(graph.blocks.keys())

# Placeholders para contenido dinámico
chart_placeholder = st.empty()
metrics_placeholder = st.empty()
table_placeholder = st.empty()

# Loop de actualización
if auto_refresh:
    import time
    
    iteration = 0
    
    while True:
        # Obtener gates del Reasoner
        gates_dict = get_reasoner_gates()
        status = get_reasoner_status()
        
        # Mostrar estado en sidebar
        with status_placeholder.container():
            if "error" not in status:
                if status.get("running"):
                    st.warning(f"🔄 Evolucionando... Gen {status['generation']}")
                else:
                    st.success("✅ Listo")
                st.metric("Best Loss", f"{status.get('best_loss', 0):.4f}")
            else:
                st.error("❌ Desconectado")
        
        # Preparar datos
        data, nx_graph, activations, plan_means, gat_output, gates_array = prepare_graph_data(
            graph, gates_dict
        )
        
        # Crear figura
        fig = build_plotly_figure(
            nx_graph,
            node_names,
            plan_means,
            gates_array,
            color_by=color_by,
        )
        
        # Mostrar gráfico (con key único para evitar duplicados)
        with chart_placeholder:
            st.plotly_chart(fig, use_container_width=True, key=f"graph_chart_{iteration}")
        
        # Métricas por bloque
        with metrics_placeholder.container():
            st.subheader("📊 Métricas por Bloque")
            
            cols = st.columns(len(node_names))
            
            for i, (col, name) in enumerate(zip(cols, node_names)):
                with col:
                    st.metric(
                        name,
                        f"{plan_means[i]:.3f}",
                        delta=f"Gate: {gates_array[i]:.2f}" if gates_array is not None else None,
                    )
        
        # Tabla detallada
        with table_placeholder.container():
            st.subheader("📋 Tabla Detallada")
            
            df_data = {
                "Bloque": node_names,
                "Activación": [f"{act:.4f}" for act in activations],
                "Plan Latente": [f"{plan:.4f}" for plan in plan_means],
                "GAT Output": [f"{gat:.4f}" for gat in gat_output],
            }
            
            if gates_array is not None:
                df_data["Reasoner Gate"] = [f"{gate:.4f}" for gate in gates_array]
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Esperar antes de siguiente actualización
        iteration += 1
        time.sleep(refresh_interval)
        
        if iteration > 1000:
            iteration = 0

else:
    # Modo estático
    gates_dict = get_reasoner_gates()
    data, nx_graph, activations, plan_means, gat_output, gates_array = prepare_graph_data(
        graph, gates_dict
    )
    
    fig = build_plotly_figure(nx_graph, node_names, plan_means, gates_array, color_by=color_by)
    st.plotly_chart(fig, use_container_width=True, key="graph_chart_static")
    
    st.info("Auto-refresh deshabilitado. Habilítalo en la sidebar para ver actualizaciones.")
