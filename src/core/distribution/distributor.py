"""Cliente para compartir agentes cognitivos entre nodos remotos."""

from __future__ import annotations

import base64
import json
from typing import Iterable

import requests

from core.persistence.file_manager import memory_path, weight_path
from core.persistence.serializer import save_memory, save_weights
from core.security import get_api_headers
from core.society.society_manager import SocietyManager


class CognitiveDistributor:
    """Envía memorias y pesos de agentes a un nodo remoto mediante HTTP."""

    def __init__(
        self,
        remote_url: str,
        api_key: str,
        timeout: float = 10.0,
        share_limit: int = 50,
    ) -> None:
        self.remote_url = remote_url.rstrip("/")
        self.headers = get_api_headers(api_key)
        self.headers.setdefault("Content-Type", "application/json")
        self.timeout = timeout
        self.share_limit = max(1, share_limit)
        self.session = requests.Session()

    def share_agent(self, agent) -> bool:
        """Serializa y envía la memoria/pesos de un agente al nodo remoto."""

        save_weights(agent)
        save_memory(agent, limit=self.share_limit)

        memory_file = memory_path(agent.name)
        weights_file = weight_path(agent.name)

        try:
            episodes: list[dict[str, object]] = []
            if memory_file.exists():
                with memory_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                episodes = data[-self.share_limit :]

            encoded_weights = None
            if weights_file.exists():
                encoded_weights = base64.b64encode(weights_file.read_bytes()).decode("utf-8")

            payload = {
                "agent_name": agent.name,
                "memory": episodes,
                "weights": encoded_weights,
            }

            response = self.session.post(
                f"{self.remote_url}/share",
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover - logging side effect
            print(f"[Distributor] Error al enviar {agent.name}: {exc}")
            return False

    def share_society(self, society: SocietyManager | Iterable) -> None:
        """Envía todos los agentes de una sociedad cognitiva."""

        agents = society.agents if isinstance(society, SocietyManager) else society
        for agent in agents:
            self.share_agent(agent)
