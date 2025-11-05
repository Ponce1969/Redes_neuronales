"""Cliente HTTP para participar en aprendizaje federado."""

from __future__ import annotations

from typing import Any, Optional

import requests

from core.federation.utils import apply_weights, serialize_agent
from core.security import get_api_headers
from core.society.agent import CognitiveAgent


class FederatedClient:
    """Envía y recibe pesos federados hacia un servidor central."""

    def __init__(
        self,
        agent: Optional[CognitiveAgent],
        server_url: str,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.agent: Optional[CognitiveAgent] = agent
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = get_api_headers(api_key)
        self.headers.setdefault("Content-Type", "application/json")

    def send_weights(self, factor: float | None = None) -> dict[str, Any]:
        """Sube los pesos actuales del agente al servidor federado."""

        if self.agent is None:
            raise ValueError("El FederatedClient no tiene un agente asignado")

        payload: dict[str, Any] = {
            "agent_name": self.agent.name,
            "weights": serialize_agent(self.agent),
        }
        if factor is not None:
            payload["factor"] = float(factor)

        response = self.session.post(
            f"{self.server_url}/federate/upload",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def receive_global_weights(self) -> dict[str, Any]:
        """Descarga pesos agregados y los aplica al agente local."""

        response = self.session.get(
            f"{self.server_url}/federate/global",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        weights = payload.get("weights", {})
        if self.agent is None:
            raise ValueError("El FederatedClient no tiene un agente asignado")
        apply_weights(self.agent, weights)
        return payload
