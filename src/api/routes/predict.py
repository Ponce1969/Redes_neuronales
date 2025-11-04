"""Endpoint de predicción para la sociedad cognitiva."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import CognitiveAppState, get_app_state

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    inputs: list[float]


@router.post("/predict")
async def predict(data: PredictRequest, state: CognitiveAppState = Depends(get_app_state)) -> dict[str, object]:
    with state.lock:
        agent = np.random.choice(state.society.agents)
        X = np.array([data.inputs], dtype=np.float32)
        outputs = agent.graph.forward({"input": X})
        stored_outputs = {name: tensor.data for name, tensor in outputs.items()}
        attention_snapshot = getattr(agent.graph, "last_attention", {})
        agent.memory_system.record_experience(
            {"input": X},
            None,
            stored_outputs,
            0.0,
            attention_snapshot,
        )
        serialized = {name: value.tolist() for name, value in stored_outputs.items()}
        state.record_predict()

        return {"agent": agent.name, "output": serialized}
