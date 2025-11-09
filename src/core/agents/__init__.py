"""
Cognitive Agents System - Sistema agentivo para razonamiento y acción.

Este módulo implementa un sistema de agentes inspirado en Claude Agent SDK
que permite razonamiento cognitivo con el loop Plan-Act-Reflect.

Componentes principales:
- BaseAgent: Abstracción base para todos los agentes
- ContextAgent: Recopila información del sistema
- PlannerAgent: Genera planes de acción
- ActionAgent: Ejecuta acciones con tools
- VerifierAgent: Verifica calidad de resultados
- ReflectorAgent: Reflexiona y aprende
- CognitiveOrchestrator: Coordina el loop completo

Uso básico:
    from core.agents import CognitiveOrchestrator, create_default_orchestrator
    
    # Crear orchestrator con config por defecto
    orchestrator = create_default_orchestrator(graph, reasoner_manager)
    
    # Ejecutar loop
    result = await orchestrator.loop(
        max_iterations=3,
        goal="optimize_performance"
    )

Fase: 35 (MVP - Día 1)
Autor: Neural Core Team
"""

from core.agents.base import (
    BaseAgent,
    AgentAction,
    AgentObservation,
    AgentThought,
    AgentRegistry,
)

from core.agents.context_agent import ContextAgent
from core.agents.planner_agent import PlannerAgent
from core.agents.action_agent import ActionAgent
from core.agents.verifier_agent import VerifierAgent
from core.agents.reflector_agent import ReflectorAgent
from core.agents.orchestrator import CognitiveOrchestrator
from core.agents.memory import AgentMemory, Episode, KnowledgeEntry


# ============================================================================
# Factory Functions
# ============================================================================

def create_default_orchestrator(
    graph,
    reasoner_manager=None,
    goal: str = "optimize_performance",
    verbose: bool = True,
) -> CognitiveOrchestrator:
    """
    Crea un orchestrator con configuración por defecto.
    
    Args:
        graph: CognitiveGraphHybrid instance
        reasoner_manager: ReasonerManager instance (opcional)
        goal: Objetivo inicial del loop
        verbose: Si True, imprime logs
    
    Returns:
        CognitiveOrchestrator configurado
    """
    # Crear agentes
    context_agent = ContextAgent(
        graph=graph,
        reasoner_manager=reasoner_manager,
        verbose=verbose,
    )
    
    planner_agent = PlannerAgent(
        goal=goal,
        verbose=verbose,
    )
    
    action_agent = ActionAgent(
        tool_registry=None,  # Será implementado en Día 2
        verbose=verbose,
    )
    
    verifier_agent = VerifierAgent(
        llm_client=None,  # Será implementado en Día 2
        use_llm=False,
        verbose=verbose,
    )
    
    reflector_agent = ReflectorAgent(
        memory_system=None,  # Será implementado después
        verbose=verbose,
    )
    
    # Crear orchestrator
    orchestrator = CognitiveOrchestrator(
        context_agent=context_agent,
        planner_agent=planner_agent,
        action_agent=action_agent,
        verifier_agent=verifier_agent,
        reflector_agent=reflector_agent,
        verbose=verbose,
    )
    
    # Registrar agentes
    AgentRegistry.register(context_agent)
    AgentRegistry.register(planner_agent)
    AgentRegistry.register(action_agent)
    AgentRegistry.register(verifier_agent)
    AgentRegistry.register(reflector_agent)
    
    return orchestrator


__all__ = [
    # Base
    "BaseAgent",
    "AgentAction",
    "AgentObservation",
    "AgentThought",
    "AgentRegistry",
    
    # Agents
    "ContextAgent",
    "PlannerAgent",
    "ActionAgent",
    "VerifierAgent",
    "ReflectorAgent",
    
    # Orchestrator
    "CognitiveOrchestrator",
    
    # Memory
    "AgentMemory",
    "Episode",
    "KnowledgeEntry",
    
    # Factory
    "create_default_orchestrator",
]
