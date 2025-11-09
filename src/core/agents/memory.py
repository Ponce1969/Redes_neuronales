"""
Agent Memory System - Sistema de memoria simple para agentes.

Memoria episódica y semántica para acumular experiencia.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Episode:
    """Episodio de memoria - una ejecución completa del loop."""
    
    id: str
    timestamp: str
    context: Dict[str, Any]
    plan: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    verification: Dict[str, Any]
    reflection: Dict[str, Any]
    outcome: str  # "accept", "retry", "abort"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "context": self.context,
            "plan": self.plan,
            "observations": self.observations,
            "verification": self.verification,
            "reflection": self.reflection,
            "outcome": self.outcome,
        }


@dataclass
class KnowledgeEntry:
    """Entrada de conocimiento semántico."""
    
    key: str
    value: Any
    confidence: float
    source: str  # De qué episodio/reflexión viene
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class AgentMemory:
    """
    Sistema de memoria para agentes.
    
    Características:
    - Memoria episódica: Historial de ciclos ejecutados
    - Memoria semántica: Conocimiento acumulado
    - Persistencia en disco
    - Recuperación por similitud (básico en MVP)
    
    Uso:
        memory = AgentMemory()
        
        # Guardar episodio
        episode = Episode(...)
        memory.store_episode(episode)
        
        # Guardar conocimiento
        memory.store_knowledge("best_tool", "reasoner_evolve", confidence=0.9)
        
        # Consultar
        recent = memory.get_recent_episodes(n=5)
        knowledge = memory.get_knowledge("best_tool")
    """
    
    def __init__(self, storage_dir: str = "data/agents/memory"):
        """
        Inicializa el sistema de memoria.
        
        Args:
            storage_dir: Directorio para persistir memoria
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Memoria episódica
        self.episodes: List[Episode] = []
        
        # Memoria semántica
        self.knowledge: Dict[str, KnowledgeEntry] = {}
        
        # Estadísticas
        self.stats = {
            "episodes_count": 0,
            "knowledge_count": 0,
            "success_rate": 0.0,
        }
    
    def store_episode(self, episode: Episode):
        """
        Almacena un episodio en memoria.
        
        Args:
            episode: Episode a guardar
        """
        self.episodes.append(episode)
        self.stats["episodes_count"] = len(self.episodes)
        
        # Actualizar success rate
        successes = sum(1 for ep in self.episodes if ep.outcome == "accept")
        self.stats["success_rate"] = successes / len(self.episodes)
    
    def store_knowledge(
        self,
        key: str,
        value: Any,
        confidence: float,
        source: str = "system",
    ):
        """
        Almacena conocimiento semántico.
        
        Args:
            key: Clave del conocimiento
            value: Valor
            confidence: Confianza [0, 1]
            source: Fuente del conocimiento
        """
        entry = KnowledgeEntry(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
        )
        
        self.knowledge[key] = entry
        self.stats["knowledge_count"] = len(self.knowledge)
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """
        Obtiene los últimos N episodios.
        
        Args:
            n: Número de episodios
        
        Returns:
            Lista de episodios
        """
        return self.episodes[-n:]
    
    def get_successful_episodes(self) -> List[Episode]:
        """
        Obtiene episodios exitosos.
        
        Returns:
            Lista de episodios con outcome="accept"
        """
        return [ep for ep in self.episodes if ep.outcome == "accept"]
    
    def get_failed_episodes(self) -> List[Episode]:
        """
        Obtiene episodios fallidos.
        
        Returns:
            Lista de episodios con outcome="abort"
        """
        return [ep for ep in self.episodes if ep.outcome == "abort"]
    
    def get_knowledge(self, key: str) -> Optional[KnowledgeEntry]:
        """
        Obtiene conocimiento por clave.
        
        Args:
            key: Clave
        
        Returns:
            KnowledgeEntry o None
        """
        return self.knowledge.get(key)
    
    def get_all_knowledge(self) -> Dict[str, KnowledgeEntry]:
        """
        Obtiene todo el conocimiento.
        
        Returns:
            Dict con todo el conocimiento
        """
        return self.knowledge.copy()
    
    def save(self, filename: str = "memory.json"):
        """
        Persiste memoria a disco.
        
        Args:
            filename: Nombre del archivo
        """
        filepath = self.storage_dir / filename
        
        data = {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "knowledge": {
                key: entry.to_dict()
                for key, entry in self.knowledge.items()
            },
            "stats": self.stats,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = "memory.json"):
        """
        Carga memoria desde disco.
        
        Args:
            filename: Nombre del archivo
        """
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            return
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Cargar episodios
        self.episodes = [
            Episode(**ep_data)
            for ep_data in data.get("episodes", [])
        ]
        
        # Cargar conocimiento
        self.knowledge = {
            key: KnowledgeEntry(**entry_data)
            for key, entry_data in data.get("knowledge", {}).items()
        }
        
        # Cargar stats
        self.stats = data.get("stats", self.stats)
    
    def clear(self):
        """Limpia toda la memoria."""
        self.episodes = []
        self.knowledge = {}
        self.stats = {
            "episodes_count": 0,
            "knowledge_count": 0,
            "success_rate": 0.0,
        }
    
    def get_summary(self) -> str:
        """
        Obtiene resumen de la memoria.
        
        Returns:
            String con resumen
        """
        lines = [
            "Memoria del Agente:",
            f"  Episodios: {len(self.episodes)}",
            f"  Conocimiento: {len(self.knowledge)} entradas",
            f"  Success Rate: {self.stats['success_rate']:.1%}",
        ]
        
        if self.episodes:
            recent = self.episodes[-1]
            lines.append(f"  Último episodio: {recent.outcome} ({recent.timestamp})")
        
        return "\n".join(lines)
