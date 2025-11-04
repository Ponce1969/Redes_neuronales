"""Endpoint para gatillar intercambios evolutivos en la sociedad."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import CognitiveAppState, get_app_state

router = APIRouter(tags=["evolve"])


class EvolveRequest(BaseModel):
    exchange: bool = True
    broadcast_top: bool = False


@router.post("/evolve")
async def evolve(request: EvolveRequest, state: CognitiveAppState = Depends(get_app_state)) -> dict[str, object]:
    with state.lock:
        if request.exchange:
            state.society.channel.exchange_memories()
        if request.broadcast_top:
            state.society.channel.broadcast_top_experiences()

        last_losses = [agent.performance for agent in state.society.agents]
        return {
            "status": "ok",
            "agents": [agent.name for agent in state.society.agents],
            "performances": last_losses,
        }
