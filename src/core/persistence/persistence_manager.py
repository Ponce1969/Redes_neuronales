"""Coordinador global de persistencia para la sociedad cognitiva."""

from __future__ import annotations

from typing import Any, Optional

from core.persistence.serializer import load_memory, load_weights, save_memory, save_weights


class PersistenceManager:
    """Gestiona el guardado y recuperado de los agentes de la sociedad."""

    def __init__(self, society: Any, reasoner_manager: Optional[Any] = None) -> None:
        self.society = society
        self.reasoner_manager = reasoner_manager

    def save_all(self) -> None:
        """Guarda estado de agentes y reasoner (si existe)."""
        for agent in self.society.agents:
            save_weights(agent)
            save_memory(agent)
        
        # Guardar Reasoner si está disponible
        if self.reasoner_manager:
            try:
                self.reasoner_manager.save("data/persistence/reasoner_state")
                print("[Persistence] Reasoner guardado ✅")
            except Exception as e:
                print(f"[Persistence] Error al guardar Reasoner: {e}")
        
        print("[Persistence] Estado completo guardado ✅")

    def load_all(self) -> None:
        """Carga estado de agentes y reasoner (si existe)."""
        for agent in self.society.agents:
            load_weights(agent)
            load_memory(agent)
        
        # Cargar Reasoner si está disponible
        if self.reasoner_manager:
            try:
                self.reasoner_manager.load("data/persistence/reasoner_state")
                print("[Persistence] Reasoner cargado ✅")
            except Exception as e:
                print(f"[Persistence] Reasoner no cargado: {e}")
        
        print("[Persistence] Estado completo restaurado ✅")
