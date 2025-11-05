"""Router FastAPI para coordinar aprendizaje federado."""

from __future__ import annotations

from typing import Any, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.federation.utils import average_weights
from core.security import require_api_key

router = APIRouter(prefix="/federate", tags=["federation"])

# Lista en memoria que acumula los pesos pendientes de federar
_federated_pool: List[Tuple[dict[str, dict[str, Any]], float]] = []


class UploadModel(BaseModel):
    agent_name: str = Field(..., min_length=1)
    weights: dict[str, dict[str, Any]]
    factor: float | None = Field(default=None, ge=0)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def receive_weights(
    payload: UploadModel,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Recibe pesos de un nodo participante y los agrega al pool temporal."""

    if not payload.weights:
        raise HTTPException(status_code=400, detail="No se recibieron pesos")

    factor = 1.0 if payload.factor is None else float(payload.factor)
    _federated_pool.append((payload.weights, factor))

    return {"status": "received", "pool_size": len(_federated_pool)}


@router.get("/global")
async def send_global_weights(
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Devuelve el promedio global de los pesos acumulados y limpia el pool."""

    if not _federated_pool:
        raise HTTPException(status_code=400, detail="No hay pesos disponibles")

    averaged = average_weights(_federated_pool)
    _federated_pool.clear()
    return {"weights": averaged}
