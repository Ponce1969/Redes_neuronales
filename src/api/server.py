"""Servidor FastAPI que expone la Sociedad Cognitiva como API REST."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import evolve, feedback, predict, status

app = FastAPI(title="Cognitive API Server", version="0.1")

app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(evolve.router)
app.include_router(status.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "🧠 Cognitive Server Online"}
