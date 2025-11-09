"""
Base Agent - Abstracción para todos los agentes cognitivos.

Proporciona la interfaz común para agentes que operan sobre el sistema cognitivo.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio


@dataclass
class AgentAction:
    """Acción que un agente puede ejecutar."""
    
    tool: str  # Nombre de la herramienta
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""  # Por qué esta acción
    priority: int = 0  # Mayor = más prioritario
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "tool": self.tool,
            "params": self.params,
            "reasoning": self.reasoning,
            "priority": self.priority,
        }


@dataclass
class AgentObservation:
    """Observación del resultado de una acción."""
    
    action: AgentAction
    result: Any
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "action": self.action.to_dict(),
            "result": self.result if not hasattr(self.result, 'to_dict') else self.result.to_dict(),
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentThought:
    """Pensamiento o reflexión del agente."""
    
    content: str
    reasoning_type: str  # "planning", "analysis", "reflection", "decision"
    confidence: float = 0.0  # [0, 1]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "content": self.content,
            "reasoning_type": self.reasoning_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class BaseAgent(ABC):
    """
    Base abstracta para todos los agentes cognitivos.
    
    Cada agente implementa un rol específico en el loop agentivo:
    - ContextAgent: Recopila información
    - PlannerAgent: Crea planes de acción
    - ActionAgent: Ejecuta acciones
    - VerifierAgent: Valida resultados
    - ReflectorAgent: Aprende de experiencias
    
    Características:
    - Async/await nativo
    - Logging estructurado
    - Integración con memory
    - Telemetría de performance
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        verbose: bool = False,
    ):
        """
        Inicializa el agente.
        
        Args:
            name: Nombre del agente
            description: Descripción de su rol
            verbose: Si True, imprime logs detallados
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        
        # Telemetría
        self.call_count = 0
        self.total_time = 0.0
        self.success_count = 0
        self.error_count = 0
        
        # Historia de pensamientos
        self.thoughts: List[AgentThought] = []
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa el contexto y retorna resultado.
        
        Este es el método principal que cada agente debe implementar.
        
        Args:
            context: Diccionario con información del entorno
        
        Returns:
            Diccionario con el resultado del procesamiento
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Log interno del agente."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{self.name}] [{level}] {message}")
    
    def think(
        self,
        content: str,
        reasoning_type: str = "analysis",
        confidence: float = 0.0,
    ) -> AgentThought:
        """
        Registra un pensamiento del agente.
        
        Args:
            content: Contenido del pensamiento
            reasoning_type: Tipo de razonamiento
            confidence: Confianza [0, 1]
        
        Returns:
            AgentThought registrado
        """
        thought = AgentThought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=confidence,
        )
        
        self.thoughts.append(thought)
        self.log(f"💭 {reasoning_type}: {content}")
        
        return thought
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del agente.
        
        Returns:
            Diccionario con métricas de performance
        """
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0.0
        success_rate = self.success_count / self.call_count if self.call_count > 0 else 0.0
        
        return {
            "name": self.name,
            "description": self.description,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "thoughts_count": len(self.thoughts),
        }
    
    def reset_stats(self):
        """Resetea las estadísticas del agente."""
        self.call_count = 0
        self.total_time = 0.0
        self.success_count = 0
        self.error_count = 0
        self.thoughts = []
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"
    
    def __str__(self) -> str:
        """String legible."""
        stats = self.get_stats()
        return (
            f"{self.name}\n"
            f"  Description: {self.description}\n"
            f"  Calls: {stats['call_count']}\n"
            f"  Success Rate: {stats['success_rate']:.1%}\n"
            f"  Avg Time: {stats['avg_time']:.3f}s"
        )


class AgentRegistry:
    """
    Registry global de agentes activos.
    
    Permite descubrimiento y gestión de agentes en el sistema.
    """
    
    _agents: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, agent: BaseAgent):
        """Registra un agente."""
        cls._agents[agent.name] = agent
    
    @classmethod
    def unregister(cls, name: str):
        """Desregistra un agente."""
        if name in cls._agents:
            del cls._agents[name]
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """Obtiene un agente por nombre."""
        return cls._agents.get(name)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """Lista todos los agentes registrados."""
        return list(cls._agents.keys())
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de todos los agentes."""
        return {
            name: agent.get_stats()
            for name, agent in cls._agents.items()
        }
    
    @classmethod
    def reset_all(cls):
        """Resetea todos los agentes."""
        for agent in cls._agents.values():
            agent.reset_stats()
