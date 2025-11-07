"""Servidor FastAPI que expone la Sociedad Cognitiva como API REST."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import (
    curriculum,
    evolve,
    feedback,
    predict,
    reasoner,
    save,
    share,
    status,
    websocket_graph,
)
from core.federation import federated_router

app = FastAPI(title="Cognitive API Server", version="0.1")

app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(evolve.router)
app.include_router(status.router)
app.include_router(save.router)
app.include_router(share.router)
app.include_router(federated_router)
app.include_router(websocket_graph.router)
app.include_router(reasoner.router)
app.include_router(curriculum.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "🧠 Cognitive Server Online"}
