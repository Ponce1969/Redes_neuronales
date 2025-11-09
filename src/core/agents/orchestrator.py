"""
Cognitive Orchestrator - Loop agentivo Plan-Act-Reflect.

Coordina todos los agentes en un ciclo cognitivo completo.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.agents.context_agent import ContextAgent
from core.agents.planner_agent import PlannerAgent
from core.agents.action_agent import ActionAgent
from core.agents.verifier_agent import VerifierAgent
from core.agents.reflector_agent import ReflectorAgent


class CognitiveOrchestrator:
    """
    Orquestador del loop agentivo completo.
    
    Implementa el ciclo: Context → Plan → Act → Verify → Reflect
    
    Inspirado en:
    - Claude Agent SDK
    - ReAct (Reason + Act)
    - MCTS-style planning
    
    Características:
    - Loop asíncrono completo
    - Gestión de estado entre ciclos
    - Early stopping si se cumple objetivo
    - Logging estructurado
    - Telemetría de performance
    """
    
    def __init__(
        self,
        context_agent: ContextAgent,
        planner_agent: PlannerAgent,
        action_agent: ActionAgent,
        verifier_agent: VerifierAgent,
        reflector_agent: ReflectorAgent,
        verbose: bool = True,
    ):
        """
        Inicializa el orchestrator.
        
        Args:
            context_agent: ContextAgent instance
            planner_agent: PlannerAgent instance
            action_agent: ActionAgent instance
            verifier_agent: VerifierAgent instance
            reflector_agent: ReflectorAgent instance
            verbose: Si True, imprime logs detallados
        """
        self.context_agent = context_agent
        self.planner_agent = planner_agent
        self.action_agent = action_agent
        self.verifier_agent = verifier_agent
        self.reflector_agent = reflector_agent
        
        self.verbose = verbose
        
        # Historial del loop
        self.history: List[Dict[str, Any]] = []
        
        # Estado actual
        self.current_cycle = 0
        self.is_running = False
    
    def log(self, message: str, level: str = "INFO"):
        """Log interno del orchestrator."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [Orchestrator] [{level}] {message}")
    
    async def loop(
        self,
        max_iterations: int = 3,
        goal: str = "optimize_performance",
        early_stop: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta el loop agentivo completo.
        
        Args:
            max_iterations: Máximo de iteraciones
            goal: Objetivo del loop
            early_stop: Si True, para cuando se cumple objetivo
        
        Returns:
            Diccionario con:
            - success: Si el loop completó exitosamente
            - iterations_run: Iteraciones ejecutadas
            - final_decision: Decisión final
            - history: Historial completo de ciclos
        """
        self.log(f"\n{'='*70}")
        self.log(f"🤖 INICIANDO AGENTIC LOOP")
        self.log(f"{'='*70}")
        self.log(f"Goal: {goal}")
        self.log(f"Max iterations: {max_iterations}")
        self.log(f"Early stop: {early_stop}")
        self.log(f"{'='*70}\n")
        
        self.is_running = True
        self.current_cycle = 0
        self.history = []
        
        # Configurar goal del planner
        self.planner_agent.set_goal(goal)
        
        start_time = time.time()
        
        try:
            for iteration in range(max_iterations):
                self.current_cycle = iteration + 1
                
                self.log(f"\n{'='*70}")
                self.log(f"🌀 CICLO {self.current_cycle}/{max_iterations}")
                self.log(f"{'='*70}\n")
                
                # Ejecutar ciclo completo
                cycle_result = await self._run_cycle()
                
                # Guardar en historial
                self.history.append(cycle_result)
                
                # Verificar si debemos parar
                decision = cycle_result.get("verification", {}).get("decision")
                
                if early_stop and decision == "accept":
                    self.log(f"\n✅ Early stop: objetivo cumplido en ciclo {self.current_cycle}")
                    break
                
                elif decision == "abort":
                    self.log(f"\n❌ Loop abortado en ciclo {self.current_cycle}")
                    break
                
                # Pequeña pausa entre ciclos
                if iteration < max_iterations - 1:
                    await asyncio.sleep(0.5)
            
            # Resumen final
            total_time = time.time() - start_time
            
            self.log(f"\n{'='*70}")
            self.log(f"🎉 LOOP COMPLETADO")
            self.log(f"{'='*70}")
            self.log(f"Ciclos ejecutados: {self.current_cycle}")
            self.log(f"Tiempo total: {total_time:.2f}s")
            self.log(f"Tiempo por ciclo: {total_time/self.current_cycle:.2f}s")
            self.log(f"{'='*70}\n")
            
            return {
                "success": True,
                "iterations_run": self.current_cycle,
                "final_decision": decision,
                "history": self.history,
                "total_time": total_time,
                "goal": goal,
            }
        
        except Exception as e:
            self.log(f"❌ Error en loop: {e}", level="ERROR")
            
            return {
                "success": False,
                "error": str(e),
                "iterations_run": self.current_cycle,
                "history": self.history,
            }
        
        finally:
            self.is_running = False
    
    async def _run_cycle(self) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo: Context → Plan → Act → Verify → Reflect.
        
        Returns:
            Diccionario con resultados del ciclo
        """
        cycle_start = time.time()
        
        cycle_result = {
            "cycle": self.current_cycle,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 1️⃣ CONTEXT: Recopilar información del sistema
        self.log("📊 Fase 1/5: CONTEXT (recopilando información...)")
        context_result = await self.context_agent.process({})
        cycle_result["context"] = context_result
        
        if not context_result.get("graph"):
            self.log("⚠️  Sin contexto de grafo, abortando ciclo", level="WARNING")
            cycle_result["aborted"] = True
            return cycle_result
        
        # 2️⃣ PLAN: Generar plan de acciones
        self.log("🧠 Fase 2/5: PLAN (generando plan de acciones...)")
        plan_result = await self.planner_agent.process(context_result)
        cycle_result["plan"] = plan_result
        
        plan = plan_result.get("plan", [])
        if not plan:
            self.log("⚠️  Sin acciones en el plan, abortando ciclo", level="WARNING")
            cycle_result["aborted"] = True
            return cycle_result
        
        self.log(f"   Plan generado: {len(plan)} acciones")
        for i, action in enumerate(plan, 1):
            self.log(f"      {i}. {action.tool} (priority: {action.priority})")
        
        # 3️⃣ ACT: Ejecutar acciones
        self.log("⚡ Fase 3/5: ACT (ejecutando acciones...)")
        action_result = await self.action_agent.process({"plan": plan})
        cycle_result["action"] = action_result
        
        success_count = action_result.get("success_count", 0)
        error_count = action_result.get("error_count", 0)
        self.log(f"   Ejecutadas: {success_count} éxitos, {error_count} fallos")
        
        # 4️⃣ VERIFY: Verificar resultados
        self.log("🔍 Fase 4/5: VERIFY (verificando calidad...)")
        verify_context = {
            "observations": action_result.get("observations", []),
            "plan": plan,
        }
        verify_result = await self.verifier_agent.process(verify_context)
        cycle_result["verification"] = verify_result
        
        decision = verify_result.get("decision", "unknown")
        score = verify_result.get("score", 0.0)
        self.log(f"   Decisión: {decision.upper()} (score: {score:.2f})")
        
        # 5️⃣ REFLECT: Reflexionar y aprender
        self.log("💭 Fase 5/5: REFLECT (reflexionando...)")
        reflect_context = {
            "observations": action_result.get("observations", []),
            "verification": verify_result,
            "plan": plan,
        }
        reflect_result = await self.reflector_agent.process(reflect_context)
        cycle_result["reflection"] = reflect_result
        
        insights_count = len(reflect_result.get("insights", []))
        should_update = reflect_result.get("should_update_reasoner", False)
        self.log(f"   Insights generados: {insights_count}")
        self.log(f"   Actualizar reasoner: {should_update}")
        
        # Timing
        cycle_time = time.time() - cycle_start
        cycle_result["cycle_time"] = cycle_time
        
        self.log(f"\n⏱️  Ciclo completado en {cycle_time:.2f}s")
        
        return cycle_result
    
    def get_summary(self) -> str:
        """
        Obtiene resumen legible del loop.
        
        Returns:
            String con resumen
        """
        if not self.history:
            return "No hay historial de ciclos aún"
        
        lines = [f"Resumen del Agentic Loop ({len(self.history)} ciclos):"]
        lines.append("")
        
        for i, cycle in enumerate(self.history, 1):
            decision = cycle.get("verification", {}).get("decision", "unknown")
            score = cycle.get("verification", {}).get("score", 0.0)
            cycle_time = cycle.get("cycle_time", 0.0)
            
            lines.append(f"Ciclo {i}: {decision.upper()} (score: {score:.2f}, time: {cycle_time:.1f}s)")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas completas del loop.
        
        Returns:
            Diccionario con stats
        """
        if not self.history:
            return {"cycles": 0}
        
        # Agregar decisiones
        decisions = [
            cycle.get("verification", {}).get("decision", "unknown")
            for cycle in self.history
        ]
        
        accepts = sum(1 for d in decisions if d == "accept")
        retries = sum(1 for d in decisions if d == "retry")
        aborts = sum(1 for d in decisions if d == "abort")
        
        # Agregar scores
        scores = [
            cycle.get("verification", {}).get("score", 0.0)
            for cycle in self.history
        ]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Agregar tiempos
        times = [cycle.get("cycle_time", 0.0) for cycle in self.history]
        total_time = sum(times)
        avg_time = total_time / len(times) if times else 0.0
        
        return {
            "cycles": len(self.history),
            "accepts": accepts,
            "retries": retries,
            "aborts": aborts,
            "avg_score": avg_score,
            "total_time": total_time,
            "avg_time_per_cycle": avg_time,
            "agents": {
                "context": self.context_agent.get_stats(),
                "planner": self.planner_agent.get_stats(),
                "action": self.action_agent.get_stats(),
                "verifier": self.verifier_agent.get_stats(),
                "reflector": self.reflector_agent.get_stats(),
            },
        }
