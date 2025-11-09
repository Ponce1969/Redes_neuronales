"""
Context Agent - Recopilación y síntesis de contexto cognitivo.

Recopila información del grafo, reasoner, curriculum y benchmark
para proporcionar contexto rico a otros agentes.
"""

import time
from typing import Dict, Any, Optional
from core.agents.base import BaseAgent, AgentThought
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.reasoning.reasoner_manager import ReasonerManager


class ContextAgent(BaseAgent):
    """
    Agente especializado en recopilación de contexto.
    
    Responsabilidades:
    - Extraer estado del grafo cognitivo
    - Obtener métricas del reasoner
    - Consultar historial de curriculum
    - Sintetizar información relevante
    
    Características:
    - Acceso directo a componentes core
    - Síntesis inteligente de información
    - Detección de anomalías
    - Priorización de información relevante
    """
    
    def __init__(
        self,
        graph: CognitiveGraphHybrid,
        reasoner_manager: Optional[ReasonerManager] = None,
        verbose: bool = False,
    ):
        """
        Inicializa el ContextAgent.
        
        Args:
            graph: CognitiveGraphHybrid instance
            reasoner_manager: ReasonerManager instance (opcional)
            verbose: Si True, imprime logs
        """
        super().__init__(
            name="ContextAgent",
            description="Recopila y sintetiza contexto del sistema cognitivo",
            verbose=verbose,
        )
        
        self.graph = graph
        self.reasoner_manager = reasoner_manager
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recopila contexto completo del sistema.
        
        Args:
            context: Contexto adicional (puede estar vacío)
        
        Returns:
            Diccionario con contexto completo:
            - graph_state: Estado del grafo
            - reasoner_state: Estado del reasoner
            - curriculum_state: Estado del curriculum (si disponible)
            - system_health: Salud general del sistema
            - recommendations: Recomendaciones basadas en contexto
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            self.think(
                "Iniciando recopilación de contexto del sistema cognitivo",
                reasoning_type="planning",
                confidence=1.0,
            )
            
            # 1. Estado del grafo
            graph_context = await self._gather_graph_context()
            
            # 2. Estado del reasoner
            reasoner_context = await self._gather_reasoner_context()
            
            # 3. Síntesis
            synthesis = await self._synthesize_context(
                graph_context,
                reasoner_context,
            )
            
            # 4. Detección de issues
            issues = await self._detect_issues(synthesis)
            
            result = {
                "graph": graph_context,
                "reasoner": reasoner_context,
                "synthesis": synthesis,
                "issues": issues,
                "timestamp": time.time(),
            }
            
            self.success_count += 1
            self.total_time += time.time() - start_time
            
            self.think(
                f"Contexto recopilado: {len(graph_context['blocks'])} bloques, "
                f"{len(issues)} issues detectados",
                reasoning_type="analysis",
                confidence=0.9,
            )
            
            return result
        
        except Exception as e:
            self.error_count += 1
            self.log(f"Error recopilando contexto: {e}", level="ERROR")
            
            return {
                "error": str(e),
                "success": False,
            }
    
    async def _gather_graph_context(self) -> Dict[str, Any]:
        """Recopila contexto del grafo cognitivo."""
        self.log("Recopilando contexto del grafo...")
        
        # Información básica
        blocks = list(self.graph.blocks.keys())
        
        # Estado de bloques
        block_states = {}
        for name, block in self.graph.blocks.items():
            block_states[name] = {
                "input_size": block.input_size,
                "hidden_size": block.hidden_size,
                "output_size": block.output_size,
                "name": name,
            }
        
        # Conexiones
        connections = []
        if hasattr(self.graph, 'graph_structure'):
            for source, targets in self.graph.graph_structure.items():
                for target in targets:
                    connections.append({"from": source, "to": target})
        
        # Gates recientes (si disponibles)
        last_gates = None
        if hasattr(self.graph, 'last_gates') and self.graph.last_gates is not None:
            last_gates = self.graph.last_gates.tolist()
        
        return {
            "blocks": blocks,
            "n_blocks": len(blocks),
            "block_states": block_states,
            "connections": connections,
            "n_connections": len(connections),
            "last_gates": last_gates,
        }
    
    async def _gather_reasoner_context(self) -> Dict[str, Any]:
        """Recopila contexto del reasoner."""
        if self.reasoner_manager is None:
            return {"available": False}
        
        self.log("Recopilando contexto del reasoner...")
        
        try:
            # Estado del reasoner
            status = self.reasoner_manager.status()
            
            # Historial de gates
            gates_history = self.reasoner_manager.gates_history[-10:] if self.reasoner_manager.gates_history else []
            
            return {
                "available": True,
                "status": status,
                "reasoner_state": {
                    "available": True,
                    "n_inputs": self.reasoner_manager.reasoner.n_inputs,
                    "n_hidden": self.reasoner_manager.reasoner.n_hidden,
                    "n_blocks": self.reasoner_manager.reasoner.n_blocks,
                },
                "gates_history": gates_history,
            }
        
        except Exception as e:
            self.log(f"Error obteniendo reasoner context: {e}", level="WARNING")
            return {
                "available": True,
                "error": str(e),
            }
    
    async def _synthesize_context(
        self,
        graph_ctx: Dict[str, Any],
        reasoner_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sintetiza contextos en información de alto nivel.
        
        Args:
            graph_ctx: Contexto del grafo
            reasoner_ctx: Contexto del reasoner
        
        Returns:
            Síntesis con insights
        """
        self.log("Sintetizando contexto...")
        
        synthesis = {
            "system_state": "operational",
            "complexity_level": graph_ctx["n_blocks"],
            "connectivity": graph_ctx["n_connections"],
        }
        
        # Análisis de gates
        if graph_ctx["last_gates"]:
            import numpy as np
            gates = np.array(graph_ctx["last_gates"])
            
            synthesis["gate_analysis"] = {
                "mean": float(np.mean(gates)),
                "std": float(np.std(gates)),
                "max": float(np.max(gates)),
                "min": float(np.min(gates)),
                "active_blocks": int(np.sum(gates > 0.1)),
                "dominant_block": int(np.argmax(gates)),
            }
        
        # Estado del reasoner
        if reasoner_ctx.get("available"):
            synthesis["reasoner_active"] = True
            synthesis["reasoner_mode"] = reasoner_ctx.get("mode", "unknown")
        else:
            synthesis["reasoner_active"] = False
        
        return synthesis
    
    async def _detect_issues(self, synthesis: Dict[str, Any]) -> list:
        """
        Detecta posibles issues en el sistema.
        
        Args:
            synthesis: Síntesis del contexto
        
        Returns:
            Lista de issues detectados
        """
        issues = []
        
        # Issue: Poca activación de gates
        if "gate_analysis" in synthesis:
            if synthesis["gate_analysis"]["active_blocks"] < 2:
                issues.append({
                    "type": "low_gate_activation",
                    "severity": "warning",
                    "message": f"Solo {synthesis['gate_analysis']['active_blocks']} bloques activos",
                    "recommendation": "Considerar ajustar threshold o revisar reasoner",
                })
            
            # Issue: Alta varianza en gates
            if synthesis["gate_analysis"]["std"] > 0.4:
                issues.append({
                    "type": "high_gate_variance",
                    "severity": "info",
                    "message": f"Alta varianza en gates (std={synthesis['gate_analysis']['std']:.3f})",
                    "recommendation": "Puede indicar incertidumbre del reasoner",
                })
        
        # Issue: Reasoner no disponible
        if not synthesis.get("reasoner_active"):
            issues.append({
                "type": "reasoner_unavailable",
                "severity": "error",
                "message": "ReasonerManager no está disponible",
                "recommendation": "Inicializar reasoner antes de operaciones cognitivas",
            })
        
        return issues
    
    def get_summary(self) -> str:
        """
        Obtiene resumen legible del último contexto.
        
        Returns:
            String con resumen
        """
        if not self.thoughts:
            return "No hay contexto recopilado aún"
        
        last_thought = self.thoughts[-1]
        return f"[{last_thought.timestamp}] {last_thought.content}"
