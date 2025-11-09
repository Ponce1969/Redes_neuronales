"""
Planner Agent - Planificación de acciones cognitivas.

Crea planes de acción basados en el contexto recopilado.
"""

import time
from typing import Dict, Any, List
from core.agents.base import BaseAgent, AgentAction


class PlannerAgent(BaseAgent):
    """
    Agente especializado en planificación.
    
    Responsabilidades:
    - Analizar contexto del sistema
    - Generar planes de acción
    - Priorizar acciones
    - Adaptar estrategia según objetivos
    
    Características:
    - Rule-based planning (MVP)
    - Priorización inteligente
    - Goal-oriented
    - Extensible a LLM-based planning
    """
    
    def __init__(
        self,
        goal: str = "optimize_performance",
        verbose: bool = False,
    ):
        """
        Inicializa el PlannerAgent.
        
        Args:
            goal: Objetivo del planner ('optimize_performance', 'explore', 'learn')
            verbose: Si True, imprime logs
        """
        super().__init__(
            name="PlannerAgent",
            description=f"Planifica acciones para objetivo: {goal}",
            verbose=verbose,
        )
        
        self.goal = goal
        
        # Estrategias disponibles
        self.strategies = {
            "optimize_performance": self._plan_optimization,
            "explore": self._plan_exploration,
            "learn": self._plan_learning,
            "diagnose": self._plan_diagnosis,
        }
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera plan de acciones basado en contexto.
        
        Args:
            context: Contexto del sistema (de ContextAgent)
        
        Returns:
            Diccionario con:
            - plan: Lista de AgentActions
            - reasoning: Justificación del plan
            - confidence: Confianza en el plan [0, 1]
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            self.think(
                f"Iniciando planificación con objetivo: {self.goal}",
                reasoning_type="planning",
                confidence=0.8,
            )
            
            # Seleccionar estrategia
            strategy = self.strategies.get(self.goal, self._plan_optimization)
            
            # Generar plan
            plan = await strategy(context)
            
            # Priorizar acciones
            plan = self._prioritize_actions(plan, context)
            
            # Calcular confianza
            confidence = self._calculate_confidence(plan, context)
            
            reasoning = self._generate_reasoning(plan, context)
            
            result = {
                "plan": plan,
                "reasoning": reasoning,
                "confidence": confidence,
                "goal": self.goal,
            }
            
            self.success_count += 1
            self.total_time += time.time() - start_time
            
            self.think(
                f"Plan generado con {len(plan)} acciones (confidence: {confidence:.2f})",
                reasoning_type="decision",
                confidence=confidence,
            )
            
            return result
        
        except Exception as e:
            self.error_count += 1
            self.log(f"Error generando plan: {e}", level="ERROR")
            
            return {
                "plan": [],
                "error": str(e),
                "success": False,
            }
    
    async def _plan_optimization(self, context: Dict[str, Any]) -> List[AgentAction]:
        """
        Genera plan para optimizar performance.
        
        Args:
            context: Contexto del sistema
        
        Returns:
            Lista de acciones
        """
        self.log("Generando plan de optimización...")
        
        actions = []
        
        # Analizar issues
        issues = context.get("issues", [])
        
        # 1. Resolver issues críticos primero
        for issue in issues:
            if issue["severity"] == "error":
                if issue["type"] == "reasoner_unavailable":
                    actions.append(AgentAction(
                        tool="reasoner_init",
                        params={},
                        reasoning="Inicializar reasoner para operación básica",
                        priority=10,
                    ))
        
        # 2. Optimizar reasoner si está disponible
        reasoner_ctx = context.get("reasoner", {})
        if reasoner_ctx.get("available"):
            # Evolucionar si no hay evolución reciente
            actions.append(AgentAction(
                tool="reasoner_evolve",
                params={"generations": 5, "population": 10},
                reasoning="Evolucionar reasoner para mejorar gates",
                priority=8,
            ))
        
        # 3. Ejecutar benchmark si tenemos reasoner funcional
        if reasoner_ctx.get("available"):
            actions.append(AgentAction(
                tool="benchmark_quick",
                params={"config": "curriculum_fast", "n_runs": 3},
                reasoning="Evaluar performance actual",
                priority=5,
            ))
        
        # 4. Analizar gates si hay anomalías
        synthesis = context.get("synthesis", {})
        if "gate_analysis" in synthesis:
            if synthesis["gate_analysis"]["active_blocks"] < 2:
                actions.append(AgentAction(
                    tool="analyze_gates",
                    params={},
                    reasoning="Analizar por qué pocos bloques están activos",
                    priority=6,
                ))
        
        return actions
    
    async def _plan_exploration(self, context: Dict[str, Any]) -> List[AgentAction]:
        """
        Genera plan para explorar el espacio de configuraciones.
        
        Args:
            context: Contexto del sistema
        
        Returns:
            Lista de acciones
        """
        self.log("Generando plan de exploración...")
        
        actions = [
            AgentAction(
                tool="graph_analyze",
                params={"depth": "full"},
                reasoning="Analizar estructura completa del grafo",
                priority=8,
            ),
            AgentAction(
                tool="reasoner_modes_compare",
                params={"modes": ["softmax", "topk", "threshold"]},
                reasoning="Comparar diferentes modos de reasoner",
                priority=7,
            ),
            AgentAction(
                tool="curriculum_test",
                params={"tasks": ["identity", "xor", "parity"]},
                reasoning="Probar tareas de curriculum básicas",
                priority=6,
            ),
        ]
        
        return actions
    
    async def _plan_learning(self, context: Dict[str, Any]) -> List[AgentAction]:
        """
        Genera plan para aprendizaje del reasoner.
        
        Args:
            context: Contexto del sistema
        
        Returns:
            Lista de acciones
        """
        self.log("Generando plan de aprendizaje...")
        
        actions = [
            AgentAction(
                tool="curriculum_start",
                params={"preset": "standard"},
                reasoning="Iniciar curriculum learning estándar",
                priority=10,
            ),
            AgentAction(
                tool="curriculum_monitor",
                params={"interval": 5},
                reasoning="Monitorear progreso del curriculum",
                priority=8,
            ),
        ]
        
        return actions
    
    async def _plan_diagnosis(self, context: Dict[str, Any]) -> List[AgentAction]:
        """
        Genera plan para diagnosticar problemas.
        
        Args:
            context: Contexto del sistema
        
        Returns:
            Lista de acciones
        """
        self.log("Generando plan de diagnóstico...")
        
        actions = [
            AgentAction(
                tool="system_health_check",
                params={},
                reasoning="Verificar salud general del sistema",
                priority=10,
            ),
            AgentAction(
                tool="graph_validate",
                params={},
                reasoning="Validar integridad del grafo",
                priority=9,
            ),
            AgentAction(
                tool="reasoner_test",
                params={},
                reasoning="Probar funcionalidad del reasoner",
                priority=8,
            ),
        ]
        
        return actions
    
    def _prioritize_actions(
        self,
        actions: List[AgentAction],
        context: Dict[str, Any],
    ) -> List[AgentAction]:
        """
        Prioriza acciones según contexto.
        
        Args:
            actions: Lista de acciones
            context: Contexto del sistema
        
        Returns:
            Lista de acciones ordenada por prioridad
        """
        # Ajustar prioridades según issues
        issues = context.get("issues", [])
        error_count = sum(1 for i in issues if i["severity"] == "error")
        
        if error_count > 0:
            # Aumentar prioridad de acciones de diagnóstico
            for action in actions:
                if "test" in action.tool or "check" in action.tool or "validate" in action.tool:
                    action.priority += 2
        
        # Ordenar por prioridad (mayor primero)
        return sorted(actions, key=lambda a: a.priority, reverse=True)
    
    def _calculate_confidence(
        self,
        plan: List[AgentAction],
        context: Dict[str, Any],
    ) -> float:
        """
        Calcula confianza en el plan.
        
        Args:
            plan: Plan generado
            context: Contexto del sistema
        
        Returns:
            Confianza [0, 1]
        """
        if not plan:
            return 0.0
        
        # Factores de confianza
        base_confidence = 0.7
        
        # Bonus por coherencia del plan
        coherence_bonus = 0.1 if len(plan) >= 2 else 0.0
        
        # Penalty por issues no resueltos
        issues = context.get("issues", [])
        unresolved_penalty = len(issues) * 0.05
        
        confidence = base_confidence + coherence_bonus - unresolved_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(
        self,
        plan: List[AgentAction],
        context: Dict[str, Any],
    ) -> str:
        """
        Genera explicación del plan.
        
        Args:
            plan: Plan generado
            context: Contexto del sistema
        
        Returns:
            String con razonamiento
        """
        if not plan:
            return "No se generaron acciones (contexto insuficiente)"
        
        lines = [f"Plan generado para objetivo '{self.goal}':"]
        
        for i, action in enumerate(plan, 1):
            lines.append(f"  {i}. {action.tool} (priority {action.priority})")
            lines.append(f"     Razón: {action.reasoning}")
        
        issues = context.get("issues", [])
        if issues:
            lines.append(f"\nIssues detectados: {len(issues)}")
            for issue in issues[:3]:  # Top 3
                lines.append(f"  - {issue['type']}: {issue['message']}")
        
        return "\n".join(lines)
    
    def set_goal(self, goal: str):
        """
        Cambia el objetivo del planner.
        
        Args:
            goal: Nuevo objetivo
        """
        if goal not in self.strategies:
            raise ValueError(
                f"Goal '{goal}' no válido. Disponibles: {list(self.strategies.keys())}"
            )
        
        self.goal = goal
        self.description = f"Planifica acciones para objetivo: {goal}"
        self.log(f"Objetivo cambiado a: {goal}")
