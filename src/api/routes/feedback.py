"""Endpoint de retroalimentación para los agentes cognitivos."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import CognitiveAppState, get_app_state

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    agent_name: str
    inputs: list[float]
    target: list[float]


@router.post("/feedback")
async def feedback(data: FeedbackRequest, state: CognitiveAppState = Depends(get_app_state)) -> dict[str, object]:
    with state.lock:
        agent = next((a for a in state.society.agents if a.name == data.agent_name), None)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agente no encontrado")

        X = np.array([data.inputs], dtype=np.float32)
        Y = np.array([data.target], dtype=np.float32)
        loss = agent.train_once(X, Y)

        state.record_feedback(loss)
        return {"agent": agent.name, "loss": float(loss)}
