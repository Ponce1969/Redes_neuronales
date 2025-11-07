"""WebSocket endpoint to stream the cognitive graph state in real time."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, WebSocket

from api.dependencies import get_app_state
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.pyg_bridge import CognitiveGraphAdapter

router = APIRouter(tags=["graph-stream"])


async def _build_payload(graph: CognitiveGraphHybrid) -> Dict[str, Any]:
    try:
        adapter = CognitiveGraphAdapter(graph)
        data = adapter.to_pyg()
    except Exception:  # pragma: no cover - fall back when graph not ready
        return {"nodes": [], "edges": []}

    activations = data.x[:, 0].detach().cpu().numpy().tolist()
    plans = data.x[:, 1].detach().cpu().numpy().tolist()
    nodes = [
        {
            "name": name,
            "activation": activations[idx],
            "z_plan": plans[idx],
        }
        for idx, name in enumerate(data.node_names)
    ]
    edges = data.edge_index.detach().cpu().t().tolist()
    return {"nodes": nodes, "edges": edges}


@router.websocket("/ws/graph_state")
async def websocket_graph_state(websocket: WebSocket) -> None:
    await websocket.accept()

    while True:
        state = get_app_state()
        payload: Dict[str, Any]
        with state.lock:
            if not state.society.agents:
                payload = {"nodes": [], "edges": []}
            else:
                agent = state.society.agents[0]
                graph = getattr(agent, "graph", None)
                if graph is None:
                    payload = {"nodes": [], "edges": []}
                else:
                    payload = await _build_payload(graph)

        await websocket.send_text(json.dumps(payload))
        await asyncio.sleep(2.0)
