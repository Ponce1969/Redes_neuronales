"""Endpoints de persistencia para el servidor cognitivo."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import CognitiveAppState, get_app_state, get_persistence_manager
from core.persistence.persistence_manager import PersistenceManager

router = APIRouter(tags=["persistence"])


@router.post("/save")
async def save_state(
    state: CognitiveAppState = Depends(get_app_state),
    persistence: PersistenceManager = Depends(get_persistence_manager),
) -> dict[str, str]:
    """Guarda el estado completo de la sociedad cognitiva."""

    with state.lock:
        persistence.save_all()
    return {"status": "saved"}
