"""Endpoint de estado para la sociedad cognitiva."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import CognitiveAppState, get_app_state

router = APIRouter(tags=["status"])


@router.get("/status")
async def status(state: CognitiveAppState = Depends(get_app_state)) -> dict[str, object]:
    with state.lock:
        agents_info = []
        for agent in state.society.agents:
            losses = agent.graph.monitor.loss_history[-5:]
            agents_info.append(
                {
                    "agent": agent.name,
                    "last_loss": float(losses[-1]) if losses else None,
                    "memory_size": len(agent.memory_system.memory),
                    "performance": float(agent.performance),
                }
            )

        return {
            "agents": agents_info,
            "predict_calls": state.predict_calls,
            "feedback_calls": state.feedback_calls,
        }
