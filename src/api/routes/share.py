"""Endpoints para compartir experiencias entre nodos cognitivos."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import CognitiveAppState, get_app_state
from core.distribution.receiver import apply_shared_payload
from core.security import require_api_key

router = APIRouter(tags=["distribution"])


class SharePayload(BaseModel):
    agent_name: str
    memory: list[dict[str, Any]] = []
    weights: str | None = None


@router.post("/share")
async def share_agent(
    payload: SharePayload,
    state: CognitiveAppState = Depends(get_app_state),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Recibe experiencias de otro nodo y las aplica al agente destino."""

    with state.lock:
        agent = next((a for a in state.society.agents if a.name == payload.agent_name), None)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agente no encontrado")

        result = apply_shared_payload(agent, payload.model_dump())
        return {"status": "ok", **result}
