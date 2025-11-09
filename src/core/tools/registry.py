"""
Tool Registry - Registry centralizado de herramientas.

Gestiona el descubrimiento y ejecución de tools.
"""

from typing import Dict, Optional, List, Any
from core.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """
    Registry centralizado de herramientas.
    
    Características:
    - Registro dinámico de tools
    - Ejecución con parámetros
    - Telemetría agregada
    - Descubrimiento de tools disponibles
    """
    
    def __init__(self):
        """Inicializa el registry vacío."""
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """
        Registra un tool.
        
        Args:
            tool: BaseTool instance
        """
        self._tools[tool.name] = tool
    
    def unregister(self, name: str):
        """
        Desregistra un tool.
        
        Args:
            name: Nombre del tool
        """
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Obtiene un tool por nombre.
        
        Args:
            name: Nombre del tool
        
        Returns:
            BaseTool o None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        Lista todos los tools disponibles.
        
        Returns:
            Lista de nombres
        """
        return list(self._tools.keys())
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Ejecuta un tool por nombre.
        
        Args:
            tool_name: Nombre del tool
            **kwargs: Parámetros para el tool
        
        Returns:
            ToolResult
        """
        tool = self.get(tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' no encontrado. Disponibles: {self.list_tools()}",
            )
        
        return await tool.run(**kwargs)
    
    def describe_all(self) -> List[Dict[str, Any]]:
        """
        Describe todos los tools.
        
        Returns:
            Lista de descripciones
        """
        return [tool.describe() for tool in self._tools.values()]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene estadísticas de todos los tools.
        
        Returns:
            Dict {tool_name: stats}
        """
        return {
            name: tool.get_stats()
            for name, tool in self._tools.items()
        }
    
    def __len__(self) -> int:
        """Número de tools registrados."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Verifica si un tool está registrado."""
        return name in self._tools
