"""
Evaluador de Reasoner en tareas de curriculum learning.

Integra con el grafo cognitivo real para evaluación end-to-end.
"""

import numpy as np
from typing import Dict, List, Optional
from core.curriculum.metrics import CognitiveMetrics


class CurriculumEvaluator:
    """
    Evaluador que mide el performance del Reasoner en tareas específicas.
    
    Características:
    - Integración con CognitiveGraphHybrid
    - Métricas avanzadas automáticas
    - Tracking de historial de gates
    """
    
    def __init__(self, graph, mode: str = "softmax"):
        """
        Inicializa el evaluador.
        
        Args:
            graph: Instancia de CognitiveGraphHybrid
            mode: Modo de gating ("softmax", "topk", "threshold")
        """
        self.graph = graph
        self.mode = mode
    
    def evaluate(
        self,
        reasoner,
        X: np.ndarray,
        Y: np.ndarray,
        track_gates: bool = True,
    ) -> Dict[str, float]:
        """
        Evalúa el Reasoner en un dataset completo.
        
        Args:
            reasoner: Instancia del Reasoner
            X: Inputs (samples, features)
            Y: Targets (samples, outputs)
            track_gates: Si True, guarda historial de gates
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        predictions = []
        gates_history = [] if track_gates else None
        
        for x, y_true in zip(X, Y):
            # Forward con Reasoner
            try:
                out_map = self.graph.forward_with_reasoner(
                    {"sensor": x}, 
                    reasoner, 
                    mode=self.mode
                )
                
                # Obtener predicción (último bloque)
                y_pred = list(out_map.values())[-1]
                
                # Convertir a numpy si es Tensor
                if hasattr(y_pred, 'data'):
                    y_pred = y_pred.data
                
                predictions.append(y_pred.flatten())
                
                # Guardar gates si está trackeando
                if track_gates and hasattr(self.graph, 'last_gates'):
                    gates_history.append(self.graph.last_gates.copy())
            
            except Exception as e:
                # Si falla, predecir zeros (penaliza fuerte)
                predictions.append(np.zeros_like(y_true).flatten())
                if track_gates:
                    gates_history.append(np.zeros(len(self.graph.blocks)))
        
        # Convertir a arrays
        predictions = np.array(predictions)
        targets = np.array([y.flatten() for y in Y])
        
        # Calcular todas las métricas
        metrics = CognitiveMetrics.compute_all(
            predictions,
            targets,
            gates_history if track_gates else None,
        )
        
        return metrics
    
    def evaluate_single(
        self,
        reasoner,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Evalúa una sola muestra (útil para evolución rápida).
        
        Args:
            reasoner: Instancia del Reasoner
            x: Input único (features,)
            y: Target único (outputs,)
        
        Returns:
            MSE loss para esta muestra
        """
        try:
            out_map = self.graph.forward_with_reasoner(
                {"sensor": x},
                reasoner,
                mode=self.mode
            )
            
            y_pred = list(out_map.values())[-1]
            
            if hasattr(y_pred, 'data'):
                y_pred = y_pred.data
            
            return float(np.mean((y_pred.flatten() - y.flatten()) ** 2))
        
        except Exception:
            # Penalización máxima en caso de error
            return 1.0
    
    def batch_evaluate(
        self,
        reasoners: List,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> List[float]:
        """
        Evalúa múltiples Reasoners en paralelo (útil para evolución).
        
        Args:
            reasoners: Lista de Reasoners a evaluar
            X: Inputs compartidos
            Y: Targets compartidos
        
        Returns:
            Lista de losses (uno por reasoner)
        """
        losses = []
        
        for reasoner in reasoners:
            metrics = self.evaluate(reasoner, X, Y, track_gates=False)
            losses.append(metrics['mse_loss'])
        
        return losses


def evaluate_reasoner_on_task(
    graph,
    reasoner,
    X: np.ndarray,
    Y: np.ndarray,
    mode: str = "softmax",
) -> float:
    """
    Función helper rápida para evaluar un Reasoner.
    
    Args:
        graph: CognitiveGraphHybrid
        reasoner: Reasoner a evaluar
        X: Inputs
        Y: Targets
        mode: Modo de gating
    
    Returns:
        MSE loss promedio
    """
    evaluator = CurriculumEvaluator(graph, mode=mode)
    metrics = evaluator.evaluate(reasoner, X, Y, track_gates=False)
    return metrics['mse_loss']
