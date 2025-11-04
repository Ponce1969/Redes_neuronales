"""Coordinador global de persistencia para la sociedad cognitiva."""

from __future__ import annotations

from typing import Any

from core.persistence.serializer import load_memory, load_weights, save_memory, save_weights


class PersistenceManager:
    """Gestiona el guardado y recuperado de los agentes de la sociedad."""

    def __init__(self, society: Any) -> None:
        self.society = society

    def save_all(self) -> None:
        for agent in self.society.agents:
            save_weights(agent)
            save_memory(agent)
        print("[Persistence] Estado completo guardado ✅")

    def load_all(self) -> None:
        for agent in self.society.agents:
            load_weights(agent)
            load_memory(agent)
        print("[Persistence] Estado completo restaurado ✅")
