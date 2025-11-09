"""
Verifier Agent - Verificación y validación de resultados.

Valida la calidad de las acciones ejecutadas, con soporte para LLM-as-Judge.
"""

import time
from typing import Dict, Any, List, Optional
from core.agents.base import BaseAgent


class VerifierAgent(BaseAgent):
    """
    Agente especializado en verificación de resultados.
    
    Responsabilidades:
    - Verificar calidad de resultados
    - Aplicar reglas de validación
    - Calcular scores de confianza
    - Decidir accept/retry/abort
    - (Futuro) LLM-as-Judge para evaluación fuzzy
    
    Características:
    - Rule-based verification (MVP)
    - Fuzzy logic scoring
    - Multi-criteria evaluation
    - Extensible a LLM verification
    """
    
    def __init__(
        self,
        llm_client=None,
        use_llm: bool = False,
        acceptance_threshold: float = 0.75,
        verbose: bool = False,
    ):
        """
        Inicializa el VerifierAgent.
        
        Args:
            llm_client: Cliente LLM (Gemini, DeepSeek, etc.) - Día 2
            use_llm: Si True, usa LLM-as-Judge
            acceptance_threshold: Umbral para aceptar resultados [0, 1]
            verbose: Si True, imprime logs
        """
        super().__init__(
            name="VerifierAgent",
            description="Verifica calidad de resultados con reglas fuzzy",
            verbose=verbose,
        )
        
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.acceptance_threshold = acceptance_threshold
        
        # Pesos para scoring (ajustables)
        self.weights = {
            "performance": 0.4,
            "stability": 0.3,
            "efficiency": 0.2,
            "novelty": 0.1,
        }
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica resultados de acciones ejecutadas.
        
        Args:
            context: Debe contener 'observations' de ActionAgent
        
        Returns:
            Diccionario con:
            - decision: "accept", "retry", "abort"
            - score: Score global [0, 1]
            - scores_detail: Scores por criterio
            - reasoning: Justificación de la decisión
            - recommendations: Recomendaciones de mejora
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            observations = context.get("observations", [])
            
            if not observations:
                self.log("No hay observaciones para verificar", level="WARNING")
                return {
                    "decision": "abort",
                    "score": 0.0,
                    "reasoning": "No hay observaciones para evaluar",
                }
            
            self.think(
                f"Verificando {len(observations)} observaciones",
                reasoning_type="analysis",
                confidence=0.85,
            )
            
            # Evaluación según método
            if self.use_llm and self.llm_client is not None:
                result = await self._verify_with_llm(observations, context)
            else:
                result = await self._verify_with_rules(observations, context)
            
            self.success_count += 1
            self.total_time += time.time() - start_time
            
            self.think(
                f"Decisión: {result['decision']} (score: {result['score']:.2f})",
                reasoning_type="decision",
                confidence=result["score"],
            )
            
            return result
        
        except Exception as e:
            self.error_count += 1
            self.log(f"Error verificando resultados: {e}", level="ERROR")
            
            return {
                "decision": "abort",
                "error": str(e),
                "success": False,
            }
    
    async def _verify_with_rules(
        self,
        observations: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verificación basada en reglas fuzzy (MVP).
        
        Args:
            observations: Lista de observaciones
            context: Contexto completo
        
        Returns:
            Diccionario con verificación
        """
        self.log("Verificando con reglas fuzzy...")
        
        # Calcular scores por criterio
        scores_detail = {}
        
        # 1. Performance
        scores_detail["performance"] = self._score_performance(observations)
        
        # 2. Stability
        scores_detail["stability"] = self._score_stability(observations)
        
        # 3. Efficiency
        scores_detail["efficiency"] = self._score_efficiency(observations)
        
        # 4. Novelty
        scores_detail["novelty"] = self._score_novelty(observations)
        
        # Score global ponderado
        global_score = sum(
            scores_detail[criterion] * self.weights[criterion]
            for criterion in scores_detail
        )
        
        # Decisión
        if global_score >= self.acceptance_threshold:
            decision = "accept"
        elif global_score >= 0.5:
            decision = "retry"
        else:
            decision = "abort"
        
        # Reasoning
        reasoning = self._generate_reasoning(decision, global_score, scores_detail)
        
        # Recomendaciones
        recommendations = self._generate_recommendations(scores_detail, observations)
        
        return {
            "decision": decision,
            "score": global_score,
            "scores_detail": scores_detail,
            "reasoning": reasoning,
            "recommendations": recommendations,
            "method": "rule_based",
        }
    
    async def _verify_with_llm(
        self,
        observations: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verificación con LLM-as-Judge (Día 2).
        
        Args:
            observations: Lista de observaciones
            context: Contexto completo
        
        Returns:
            Diccionario con verificación
        """
        self.log("Verificando con LLM-as-Judge...")
        
        # TODO Día 2: Implementar LLM verification
        # Por ahora, fallback a rules
        self.log("LLM verification no disponible, usando rules", level="WARNING")
        return await self._verify_with_rules(observations, context)
    
    def _score_performance(self, observations: List[Dict[str, Any]]) -> float:
        """
        Evalúa performance basado en métricas de resultados.
        
        Args:
            observations: Observaciones
        
        Returns:
            Score [0, 1]
        """
        scores = []
        
        for obs in observations:
            if not obs.get("success"):
                scores.append(0.0)
                continue
            
            result = obs.get("result", {})
            
            # Si tiene loss, score inverso
            if "loss" in result:
                loss = result["loss"]
                score = max(0.0, 1.0 - loss)  # Asume loss en [0, 1]
                scores.append(score)
            
            # Si tiene accuracy, directo
            elif "accuracy" in result:
                scores.append(result["accuracy"])
            
            # Si tiene improvement
            elif "improvement" in result:
                scores.append(min(1.0, result["improvement"]))
            
            else:
                scores.append(0.7)  # Score neutro
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _score_stability(self, observations: List[Dict[str, Any]]) -> float:
        """
        Evalúa estabilidad de ejecución.
        
        Args:
            observations: Observaciones
        
        Returns:
            Score [0, 1]
        """
        # Proporción de éxitos
        successes = sum(1 for obs in observations if obs.get("success"))
        success_rate = successes / len(observations) if observations else 0.0
        
        # Bonus si no hubo reintentos
        no_retries = all(
            obs.get("metadata", {}).get("retries", 0) == 0
            for obs in observations
        )
        
        stability = success_rate * 0.8
        if no_retries:
            stability += 0.2
        
        return min(1.0, stability)
    
    def _score_efficiency(self, observations: List[Dict[str, Any]]) -> float:
        """
        Evalúa eficiencia de ejecución.
        
        Args:
            observations: Observaciones
        
        Returns:
            Score [0, 1]
        """
        # Cuenta reintentos y errores
        total_retries = sum(
            obs.get("metadata", {}).get("retries", 0)
            for obs in observations
        )
        
        errors = sum(1 for obs in observations if not obs.get("success"))
        
        # Penalty por ineficiencias
        efficiency = 1.0 - (total_retries * 0.1 + errors * 0.2)
        
        return max(0.0, efficiency)
    
    def _score_novelty(self, observations: List[Dict[str, Any]]) -> float:
        """
        Evalúa novedad/exploración.
        
        Args:
            observations: Observaciones
        
        Returns:
            Score [0, 1]
        """
        # Por ahora, score neutro
        # TODO Día 2: Evaluar si explora nuevas estrategias
        return 0.5
    
    def _generate_reasoning(
        self,
        decision: str,
        score: float,
        scores_detail: Dict[str, float],
    ) -> str:
        """
        Genera explicación de la decisión.
        
        Args:
            decision: Decisión tomada
            score: Score global
            scores_detail: Scores por criterio
        
        Returns:
            String con razonamiento
        """
        lines = [f"Decisión: {decision.upper()} (score global: {score:.2f})"]
        lines.append("\nScores por criterio:")
        
        for criterion, crit_score in sorted(scores_detail.items(), key=lambda x: x[1], reverse=True):
            weight = self.weights[criterion]
            lines.append(f"  - {criterion:15s}: {crit_score:.2f} (peso: {weight:.1f})")
        
        if decision == "accept":
            lines.append(f"\n✅ Resultados aceptables (>= {self.acceptance_threshold:.2f})")
        elif decision == "retry":
            lines.append(f"\n⚠️  Resultados sub-óptimos, recomendar reintento")
        else:
            lines.append(f"\n❌ Resultados inaceptables, abortar")
        
        return "\n".join(lines)
    
    def _generate_recommendations(
        self,
        scores_detail: Dict[str, float],
        observations: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Genera recomendaciones de mejora.
        
        Args:
            scores_detail: Scores por criterio
            observations: Observaciones
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Identificar criterios débiles
        weak_criteria = [
            criterion for criterion, score in scores_detail.items()
            if score < 0.6
        ]
        
        for criterion in weak_criteria:
            if criterion == "performance":
                recommendations.append(
                    "Mejorar performance: considerar más evolución o ajustar parámetros"
                )
            elif criterion == "stability":
                recommendations.append(
                    "Mejorar estabilidad: revisar errores y reducir reintentos"
                )
            elif criterion == "efficiency":
                recommendations.append(
                    "Mejorar eficiencia: optimizar parámetros de tools o reducir complejidad"
                )
            elif criterion == "novelty":
                recommendations.append(
                    "Explorar más: probar diferentes estrategias o herramientas"
                )
        
        # Revisar errores
        errors = [obs for obs in observations if not obs.get("success")]
        if errors:
            recommendations.append(
                f"Resolver {len(errors)} error(es) en ejecución antes de continuar"
            )
        
        if not recommendations:
            recommendations.append("Resultados satisfactorios, continuar con siguiente fase")
        
        return recommendations
