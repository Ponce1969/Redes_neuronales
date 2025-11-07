"""Streamlit dashboard streaming the cognitive graph state via WebSocket."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import websocket

WS_URL = "ws://localhost:8000/ws/graph_state"


st.set_page_config(page_title="🧠 Live Cognitive Stream", layout="wide")
st.title("🧩 Cognitive Graph – Live Stream")
st.write("Actualización automática del estado cognitivo vía WebSocket.")

if "graph_data" not in st.session_state:
    st.session_state.graph_data = {"nodes": [], "edges": []}


def on_message(_: websocket.WebSocketApp, message: str) -> None:
    st.session_state.graph_data = json.loads(message)


def on_error(_: websocket.WebSocketApp, error: Exception) -> None:
    st.session_state.graph_data = {"nodes": [], "edges": [], "error": str(error)}


def listen_ws() -> None:
    ws = websocket.WebSocketApp(WS_URL, on_message=on_message, on_error=on_error)
    ws.run_forever()


def ensure_listener_running() -> None:
    if not any(t.name == "graph-listener" for t in threading.enumerate()):
        thread = threading.Thread(target=listen_ws, name="graph-listener", daemon=True)
        thread.start()


ensure_listener_running()

placeholder_chart = st.empty()
placeholder_table = st.empty()
placeholder_status = st.info("Esperando datos en vivo…")


def build_figure(data: Dict[str, Any]) -> go.Figure:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        return go.Figure(layout=go.Layout(showlegend=False))

    n = len(nodes)
    theta = [2 * np.pi * i / n for i in range(n)]
    radius = 1.65
    x_nodes = [radius * np.cos(t) for t in theta]
    y_nodes = [radius * np.sin(t) for t in theta]
    z_vals = [node.get("z_plan", 0.0) for node in nodes]

    x_edges: list[float | None] = []
    y_edges: list[float | None] = []
    for src, dst in edges:
        x_edges += [x_nodes[src], x_nodes[dst], None]
        y_edges += [y_nodes[src], y_nodes[dst], None]

    edge_trace = go.Scatter(
        x=x_edges,
        y=y_edges,
        line=dict(width=2, color="rgba(150,150,150,0.4)"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=[node.get("name", "?") for node in nodes],
        hovertext=[f"z={node.get('z_plan', 0.0):.4f}" for node in nodes],
        marker=dict(
            colorscale="Viridis",
            color=z_vals,
            size=60,
            colorbar=dict(title="Z_plan", thickness=12),
            line_width=2,
        ),
    )

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        ),
    )


while True:
    graph_data = st.session_state.graph_data
    error_msg = graph_data.get("error")
    if error_msg:
        placeholder_status.error(f"Error en WebSocket: {error_msg}")
    elif graph_data.get("nodes"):
        placeholder_status.empty()
        fig = build_figure(graph_data)
        placeholder_chart.plotly_chart(fig, use_container_width=True)
        df = pd.DataFrame(graph_data["nodes"]).round(4)
        placeholder_table.dataframe(df, use_container_width=True)
    else:
        placeholder_status.info("Esperando datos en vivo…")
    time.sleep(2.0)
