"""
Base Tool - Abstracción para herramientas cognitivas.

Define la interfaz para tools que los agentes pueden ejecutar.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class ToolResult:
    """Resultado de ejecutar una herramienta."""
    
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """
    Base abstracta para todas las herramientas.
    
    Cada tool debe implementar:
    - execute(): Ejecuta la herramienta
    - describe(): Describe qué hace
    
    Características:
    - Async/await nativo
    - Error handling
    - Timeout management
    - Telemetría
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        timeout: float = 30.0,
    ):
        """
        Inicializa el tool.
        
        Args:
            name: Nombre del tool
            description: Descripción de qué hace
            timeout: Timeout en segundos
        """
        self.name = name
        self.description = description
        self.timeout = timeout
        
        # Telemetría
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_time = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Ejecuta el tool.
        
        Args:
            **kwargs: Parámetros específicos del tool
        
        Returns:
            ToolResult con resultado
        """
        pass
    
    async def run(self, **kwargs) -> ToolResult:
        """
        Wrapper que ejecuta con telemetría.
        
        Args:
            **kwargs: Parámetros del tool
        
        Returns:
            ToolResult
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            result = await self.execute(**kwargs)
            
            if result.success:
                self.success_count += 1
            else:
                self.error_count += 1
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            self.total_time += execution_time
            
            return result
        
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            self.total_time += execution_time
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
            )
    
    def describe(self) -> Dict[str, Any]:
        """
        Describe el tool.
        
        Returns:
            Diccionario con descripción
        """
        return {
            "name": self.name,
            "description": self.description,
            "timeout": self.timeout,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del tool.
        
        Returns:
            Diccionario con stats
        """
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0.0
        success_rate = self.success_count / self.call_count if self.call_count > 0 else 0.0
        
        return {
            "name": self.name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_time": self.total_time,
            "avg_time": avg_time,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"
