"""
Tool System - Sistema de herramientas para agentes cognitivos.

Este módulo implementa el tool system que permite a los agentes
ejecutar acciones sobre el sistema cognitivo.

Componentes:
- BaseTool: Abstracción base
- ToolRegistry: Registry centralizado
- Cognitive Tools: Tools específicos (reasoner, graph, curriculum, benchmark)

Uso básico:
    from core.tools import create_default_registry
    
    registry = create_default_registry(graph, reasoner_manager)
    
    # Ejecutar tool
    result = await registry.execute("graph_analyze", depth="full")

Fase: 35 (MVP - Día 1)
Autor: Neural Core Team
"""

from core.tools.base import BaseTool, ToolResult
from core.tools.registry import ToolRegistry
from core.tools.cognitive_tools import (
    ReasonerEvolveTool,
    GraphAnalyzeTool,
    CurriculumStartTool,
    BenchmarkQuickTool,
    SystemHealthCheckTool,
)


def create_default_registry(graph, reasoner_manager=None) -> ToolRegistry:
    """
    Crea un ToolRegistry con tools por defecto.
    
    Args:
        graph: CognitiveGraphHybrid instance
        reasoner_manager: ReasonerManager instance (opcional)
    
    Returns:
        ToolRegistry configurado
    """
    registry = ToolRegistry()
    
    # Registrar cognitive tools
    registry.register(GraphAnalyzeTool(graph))
    registry.register(SystemHealthCheckTool(graph, reasoner_manager))
    
    if reasoner_manager is not None:
        registry.register(ReasonerEvolveTool(reasoner_manager, graph))
    
    # Tools que no requieren instancias específicas
    registry.register(CurriculumStartTool())
    registry.register(BenchmarkQuickTool())
    
    return registry


__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    
    # Registry
    "ToolRegistry",
    "create_default_registry",
    
    # Cognitive Tools
    "ReasonerEvolveTool",
    "GraphAnalyzeTool",
    "CurriculumStartTool",
    "BenchmarkQuickTool",
    "SystemHealthCheckTool",
]
