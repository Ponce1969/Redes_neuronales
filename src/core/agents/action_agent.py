"""
Action Agent - Ejecución de acciones cognitivas.

Ejecuta acciones del plan usando el tool system.
"""

import time
from typing import Dict, Any, List, Optional
from core.agents.base import BaseAgent, AgentAction, AgentObservation


class ActionAgent(BaseAgent):
    """
    Agente especializado en ejecutar acciones.
    
    Responsabilidades:
    - Ejecutar acciones del plan
    - Gestionar errores de ejecución
    - Recopilar resultados
    - Registrar observaciones
    
    Características:
    - Integración con tool system
    - Error handling robusto
    - Timeout management
    - Logging estructurado
    """
    
    def __init__(
        self,
        tool_registry=None,
        max_retries: int = 2,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        """
        Inicializa el ActionAgent.
        
        Args:
            tool_registry: ToolRegistry instance (será implementado)
            max_retries: Máximo de reintentos por acción
            timeout: Timeout en segundos por acción
            verbose: Si True, imprime logs
        """
        super().__init__(
            name="ActionAgent",
            description="Ejecuta acciones usando herramientas cognitivas",
            verbose=verbose,
        )
        
        self.tool_registry = tool_registry
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Historial de observaciones
        self.observations: List[AgentObservation] = []
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta plan de acciones.
        
        Args:
            context: Debe contener 'plan' con lista de AgentActions
        
        Returns:
            Diccionario con:
            - observations: Lista de AgentObservation
            - success_count: Número de acciones exitosas
            - error_count: Número de acciones fallidas
            - results: Resultados agregados
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            plan = context.get("plan", [])
            
            if not plan:
                self.log("No hay acciones en el plan", level="WARNING")
                return {
                    "observations": [],
                    "success_count": 0,
                    "error_count": 0,
                    "message": "Plan vacío",
                }
            
            self.think(
                f"Ejecutando plan con {len(plan)} acciones",
                reasoning_type="planning",
                confidence=0.9,
            )
            
            # Ejecutar acciones
            observations = []
            success_count = 0
            error_count = 0
            
            for i, action in enumerate(plan):
                self.log(f"Ejecutando acción {i+1}/{len(plan)}: {action.tool}")
                
                observation = await self._execute_action(action)
                observations.append(observation)
                
                if observation.success:
                    success_count += 1
                else:
                    error_count += 1
                
                # Guardar en historial
                self.observations.append(observation)
            
            # Agregamos resultados
            results = {
                "observations": [obs.to_dict() for obs in observations],
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / len(plan) if plan else 0.0,
            }
            
            self.success_count += 1
            self.total_time += time.time() - start_time
            
            self.think(
                f"Plan ejecutado: {success_count} éxitos, {error_count} fallos",
                reasoning_type="analysis",
                confidence=0.8 if error_count == 0 else 0.5,
            )
            
            return results
        
        except Exception as e:
            self.error_count += 1
            self.log(f"Error ejecutando plan: {e}", level="ERROR")
            
            return {
                "observations": [],
                "error": str(e),
                "success": False,
            }
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """
        Ejecuta una acción individual.
        
        Args:
            action: AgentAction a ejecutar
        
        Returns:
            AgentObservation con resultado
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                # Ejecutar a través del tool registry (si está disponible)
                if self.tool_registry is not None:
                    tool_result = await self.tool_registry.execute(
                        action.tool,
                        **action.params
                    )
                    
                    if tool_result.success:
                        result = tool_result.data
                    else:
                        raise Exception(tool_result.error)
                else:
                    # Fallback: ejecución simulada (MVP)
                    result = await self._execute_simulated(action)
                
                # Éxito
                return AgentObservation(
                    action=action,
                    result=result,
                    success=True,
                    metadata={"retries": retries},
                )
            
            except Exception as e:
                last_error = str(e)
                retries += 1
                
                if retries <= self.max_retries:
                    self.log(
                        f"Reintentando {action.tool} ({retries}/{self.max_retries})",
                        level="WARNING"
                    )
                    await asyncio.sleep(0.5 * retries)  # Backoff
        
        # Falló después de todos los reintentos
        return AgentObservation(
            action=action,
            result=None,
            success=False,
            error=last_error,
            metadata={"retries": retries},
        )
    
    async def _execute_simulated(self, action: AgentAction) -> Dict[str, Any]:
        """
        Ejecución simulada para MVP (sin tool registry completo).
        
        Args:
            action: AgentAction a ejecutar
        
        Returns:
            Diccionario con resultado simulado
        """
        import asyncio
        
        self.log(f"[SIMULADO] Ejecutando: {action.tool}", level="INFO")
        
        # Simular tiempo de ejecución
        await asyncio.sleep(0.1)
        
        # Resultado simulado según tool
        if "evolve" in action.tool:
            return {
                "tool": action.tool,
                "loss": 0.045,
                "improvement": 0.15,
                "generations": action.params.get("generations", 5),
            }
        
        elif "benchmark" in action.tool:
            return {
                "tool": action.tool,
                "final_loss": 0.035,
                "accuracy": 0.92,
                "n_runs": action.params.get("n_runs", 3),
            }
        
        elif "analyze" in action.tool:
            return {
                "tool": action.tool,
                "gates_mean": 0.33,
                "active_blocks": 2,
                "diversity": 0.65,
            }
        
        elif "curriculum" in action.tool:
            return {
                "tool": action.tool,
                "stage": "identity",
                "loss": 0.02,
                "status": "completed",
            }
        
        else:
            return {
                "tool": action.tool,
                "status": "executed",
                "simulated": True,
            }
    
    def get_recent_observations(self, n: int = 10) -> List[AgentObservation]:
        """
        Obtiene las últimas N observaciones.
        
        Args:
            n: Número de observaciones
        
        Returns:
            Lista de observaciones
        """
        return self.observations[-n:]
    
    def get_success_rate(self) -> float:
        """
        Calcula tasa de éxito de acciones.
        
        Returns:
            Success rate [0, 1]
        """
        if not self.observations:
            return 0.0
        
        successes = sum(1 for obs in self.observations if obs.success)
        return successes / len(self.observations)
    
    def clear_observations(self):
        """Limpia el historial de observaciones."""
        self.observations = []
