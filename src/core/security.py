"""Utilidades de seguridad compartidas para la API cognitiva."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

API_KEY_ENV = "API_KEY"
API_KEY_HEADER = "x-api-key"


def require_api_key(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> None:
    """Dependencia de FastAPI que valida la API key entrante."""

    expected = os.getenv(API_KEY_ENV)
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key no configurada en el servidor",
        )

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
        )


def get_api_headers(api_key: str | None = None) -> dict[str, Any]:
    """Helper para construir headers autenticados desde el lado cliente."""

    key = api_key or os.getenv(API_KEY_ENV, "")
    if not key:
        raise ValueError("API key no disponible")
    return {API_KEY_HEADER: key}
