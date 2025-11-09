"""
Reflector Agent - Reflexión y aprendizaje de experiencias.

Analiza resultados y actualiza conocimiento del sistema.
"""

import time
from typing import Dict, Any, List
from core.agents.base import BaseAgent


class ReflectorAgent(BaseAgent):
    """
    Agente especializado en reflexión y aprendizaje.
    
    Responsabilidades:
    - Analizar resultados de acciones
    - Identificar patrones de éxito/fallo
    - Actualizar conocimiento
    - Generar insights
    - Proponer mejoras futuras
    
    Características:
    - Pattern recognition
    - Success/failure analysis
    - Knowledge accumulation
    - Meta-learning básico (MVP)
    """
    
    def __init__(
        self,
        memory_system=None,
        verbose: bool = False,
    ):
        """
        Inicializa el ReflectorAgent.
        
        Args:
            memory_system: Sistema de memoria (será implementado)
            verbose: Si True, imprime logs
        """
        super().__init__(
            name="ReflectorAgent",
            description="Reflexiona y aprende de experiencias pasadas",
            verbose=verbose,
        )
        
        self.memory_system = memory_system
        
        # Conocimiento acumulado
        self.insights: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflexiona sobre los resultados del ciclo.
        
        Args:
            context: Debe contener:
                - observations: Observaciones de ActionAgent
                - verification: Resultado de VerifierAgent
                - plan: Plan original
        
        Returns:
            Diccionario con:
            - insights: Nuevos insights generados
            - learnings: Aprendizajes clave
            - recommendations: Recomendaciones para futuro
            - should_update_reasoner: Si se debe actualizar reasoner
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            self.think(
                "Iniciando reflexión sobre el ciclo ejecutado",
                reasoning_type="reflection",
                confidence=0.8,
            )
            
            # Analizar observaciones
            observations = context.get("observations", [])
            verification = context.get("verification", {})
            plan = context.get("plan", [])
            
            # Generar insights
            insights = await self._generate_insights(observations, verification, plan)
            
            # Identificar patrones
            patterns = await self._identify_patterns(observations, verification)
            
            # Generar aprendizajes
            learnings = await self._extract_learnings(insights, patterns)
            
            # Recomendaciones para futuro
            recommendations = await self._generate_future_recommendations(learnings)
            
            # Decidir si actualizar reasoner
            should_update = await self._should_update_reasoner(verification, learnings)
            
            # Guardar insights
            for insight in insights:
                self.insights.append(insight)
            
            # Guardar patrones
            if patterns["success"]:
                self.success_patterns.extend(patterns["success"])
            if patterns["failure"]:
                self.failure_patterns.extend(patterns["failure"])
            
            result = {
                "insights": insights,
                "learnings": learnings,
                "recommendations": recommendations,
                "should_update_reasoner": should_update,
                "patterns": patterns,
            }
            
            self.success_count += 1
            self.total_time += time.time() - start_time
            
            self.think(
                f"Reflexión completada: {len(insights)} insights, {len(learnings)} aprendizajes",
                reasoning_type="reflection",
                confidence=0.85,
            )
            
            return result
        
        except Exception as e:
            self.error_count += 1
            self.log(f"Error en reflexión: {e}", level="ERROR")
            
            return {
                "insights": [],
                "error": str(e),
                "success": False,
            }
    
    async def _generate_insights(
        self,
        observations: List[Dict[str, Any]],
        verification: Dict[str, Any],
        plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Genera insights de alto nivel.
        
        Args:
            observations: Observaciones
            verification: Resultado de verificación
            plan: Plan original
        
        Returns:
            Lista de insights
        """
        insights = []
        
        # Insight 1: Éxito general del plan
        decision = verification.get("decision", "unknown")
        score = verification.get("score", 0.0)
        
        insights.append({
            "type": "plan_effectiveness",
            "content": f"Plan {decision} con score {score:.2f}",
            "confidence": score,
            "actionable": decision != "accept",
        })
        
        # Insight 2: Tools más efectivos
        successful_tools = [
            obs.get("action", {}).get("tool")
            for obs in observations
            if obs.get("success")
        ]
        
        if successful_tools:
            most_used = max(set(successful_tools), key=successful_tools.count)
            insights.append({
                "type": "effective_tools",
                "content": f"Tool más efectivo: {most_used}",
                "confidence": 0.7,
                "actionable": True,
            })
        
        # Insight 3: Criterios débiles
        scores_detail = verification.get("scores_detail", {})
        if scores_detail:
            weak_criterion = min(scores_detail.items(), key=lambda x: x[1])
            if weak_criterion[1] < 0.6:
                insights.append({
                    "type": "weak_criterion",
                    "content": f"Criterio débil: {weak_criterion[0]} ({weak_criterion[1]:.2f})",
                    "confidence": 1.0 - weak_criterion[1],
                    "actionable": True,
                })
        
        return insights
    
    async def _identify_patterns(
        self,
        observations: List[Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identifica patrones de éxito y fallo.
        
        Args:
            observations: Observaciones
            verification: Verificación
        
        Returns:
            Dict con patterns: {"success": [...], "failure": [...]}
        """
        patterns = {
            "success": [],
            "failure": [],
        }
        
        # Analizar acciones exitosas
        for obs in observations:
            if obs.get("success"):
                action = obs.get("action", {})
                result = obs.get("result", {})
                
                patterns["success"].append({
                    "tool": action.get("tool"),
                    "params": action.get("params"),
                    "result_metrics": self._extract_metrics(result),
                })
            else:
                action = obs.get("action", {})
                error = obs.get("error")
                
                patterns["failure"].append({
                    "tool": action.get("tool"),
                    "params": action.get("params"),
                    "error": error,
                })
        
        return patterns
    
    async def _extract_learnings(
        self,
        insights: List[Dict[str, Any]],
        patterns: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Extrae aprendizajes clave.
        
        Args:
            insights: Insights generados
            patterns: Patrones identificados
        
        Returns:
            Lista de aprendizajes
        """
        learnings = []
        
        # Learning de insights accionables
        for insight in insights:
            if insight.get("actionable") and insight.get("confidence", 0) > 0.6:
                learnings.append(insight["content"])
        
        # Learning de patrones
        if len(patterns["success"]) > len(patterns["failure"]) * 2:
            learnings.append("Las acciones generalmente son exitosas, continuar estrategia actual")
        elif len(patterns["failure"]) > len(patterns["success"]):
            learnings.append("Alta tasa de fallos, revisar estrategia o parámetros")
        
        # Learning de tools
        success_tools = [p["tool"] for p in patterns["success"]]
        if success_tools:
            unique_tools = set(success_tools)
            learnings.append(f"Tools exitosos: {', '.join(unique_tools)}")
        
        return learnings
    
    async def _generate_future_recommendations(
        self,
        learnings: List[str],
    ) -> List[str]:
        """
        Genera recomendaciones para ciclos futuros.
        
        Args:
            learnings: Aprendizajes actuales
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendaciones basadas en historial
        if len(self.success_patterns) > 10:
            recommendations.append(
                "Considerar meta-learning: suficiente historial de éxitos"
            )
        
        if len(self.failure_patterns) > 5:
            recommendations.append(
                "Revisar patrones de fallo recurrentes y ajustar estrategia"
            )
        
        # Recomendaciones de exploración
        if len(self.insights) % 5 == 0:
            recommendations.append(
                "Considerar cambio de goal para explorar nuevas estrategias"
            )
        
        # Default
        if not recommendations:
            recommendations.append(
                "Continuar con estrategia actual, acumular más experiencia"
            )
        
        return recommendations
    
    async def _should_update_reasoner(
        self,
        verification: Dict[str, Any],
        learnings: List[str],
    ) -> bool:
        """
        Decide si se debe actualizar el reasoner.
        
        Args:
            verification: Resultado de verificación
            learnings: Aprendizajes
        
        Returns:
            True si se debe actualizar
        """
        # Actualizar si score es bajo
        score = verification.get("score", 0.0)
        if score < 0.6:
            return True
        
        # Actualizar si hay muchos fallos
        if any("fallo" in learning.lower() for learning in learnings):
            return True
        
        # No actualizar si todo va bien
        if verification.get("decision") == "accept":
            return False
        
        # Default: actualizar si no estamos seguros
        return True
    
    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae métricas numéricas de un resultado.
        
        Args:
            result: Resultado de acción
        
        Returns:
            Dict con métricas
        """
        metrics = {}
        
        # Extraer métricas conocidas
        for key in ["loss", "accuracy", "improvement", "score"]:
            if key in result:
                metrics[key] = result[key]
        
        return metrics
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del conocimiento acumulado.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "total_insights": len(self.insights),
            "success_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "success_rate": (
                len(self.success_patterns) / (len(self.success_patterns) + len(self.failure_patterns))
                if (len(self.success_patterns) + len(self.failure_patterns)) > 0
                else 0.0
            ),
        }
