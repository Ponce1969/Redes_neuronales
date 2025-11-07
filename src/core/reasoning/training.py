"""
Utilidades para entrenar y evolucionar el Reasoner mediante estrategias evolutivas.

Este módulo provee funciones para optimizar el Reasoner sin necesidad de autograd,
usando mutación y selección natural sobre datasets de prueba.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    try:
        from src.core.cognitive_graph_hybrid import CognitiveGraphHybrid
        from src.core.reasoning.reasoner import Reasoner
    except ModuleNotFoundError:
        from core.cognitive_graph_hybrid import CognitiveGraphHybrid  # type: ignore
        from core.reasoning.reasoner import Reasoner  # type: ignore


def evaluate_reasoner(
    graph: Any,
    reasoner: Any,
    X: np.ndarray,
    Y: np.ndarray,
    mode: str = "softmax",
    top_k: int = 2,
) -> float:
    """
    Evalúa el Reasoner sobre un dataset sin entrenar los pesos del grafo.
    
    Calcula el MSE promedio entre las predicciones del grafo (usando el reasoner
    para gating selectivo) y los targets esperados.
    
    Args:
        graph: Instancia de CognitiveGraphHybrid
        reasoner: Instancia de Reasoner a evaluar
        X: Array de inputs (N, n_features)
        Y: Array de targets (N, n_outputs)
        mode: Modo de gating ('softmax', 'topk', 'threshold')
        top_k: Número de bloques a activar en modo 'topk'
        
    Returns:
        MSE promedio sobre el dataset
    """
    errors = []

    for x, y_true in zip(X, Y):
        # Forward normal primero para que los bloques computen last_plan/z
        _ = graph.forward({"sensor": x.tolist()})

        # Forward con reasoner
        out_map = graph.forward_with_reasoner(
            {"sensor": x.tolist()}, reasoner, mode=mode, top_k=top_k
        )

        # Obtener predicción del último bloque (asumiendo es la decisión final)
        y_pred = list(out_map.values())[-1].data.squeeze()

        # Calcular error cuadrático
        error = float((y_pred - y_true) ** 2)
        errors.append(error)

    return float(np.mean(errors))


def evolve_reasoner_on_task(
    graph: Any,
    base_reasoner: Any,
    X: np.ndarray,
    Y: np.ndarray,
    generations: int = 50,
    pop_size: int = 8,
    mutation_scale: float = 0.03,
    mode: str = "softmax",
    top_k: int = 2,
    verbose: bool = True,
) -> Tuple[Any, List[float]]:
    """
    Evoluciona el Reasoner usando estrategia evolutiva simple (1+λ).
    
    En cada generación:
    1. Genera pop_size mutantes del mejor reasoner actual
    2. Evalúa cada mutante en el dataset
    3. Si algún mutante mejora, reemplaza al padre
    4. Repite por N generaciones
    
    Args:
        graph: Instancia de CognitiveGraphHybrid
        base_reasoner: Reasoner inicial
        X: Array de inputs de entrenamiento (N, n_features)
        Y: Array de targets (N, n_outputs)
        generations: Número de generaciones evolutivas
        pop_size: Tamaño de población por generación
        mutation_scale: Magnitud de mutación gaussiana
        mode: Modo de gating del reasoner
        top_k: Número de bloques a activar en modo 'topk'
        verbose: Si imprime progreso
        
    Returns:
        Tupla (mejor_reasoner, historial_de_pérdidas)
    """
    parent = base_reasoner
    best_loss = evaluate_reasoner(graph, parent, X, Y, mode=mode, top_k=top_k)
    loss_history = [best_loss]

    if verbose:
        print(f"[Evolución] Generación 0 - Loss inicial: {best_loss:.6f}")

    for gen in range(1, generations + 1):
        # Generar mutantes
        children = [parent.mutate(scale=mutation_scale, seed=None) for _ in range(pop_size)]

        improved = False
        best_gen_loss = best_loss

        # Evaluar cada mutante
        for child in children:
            loss = evaluate_reasoner(graph, child, X, Y, mode=mode, top_k=top_k)

            if loss < best_loss:
                parent = child
                best_loss = loss
                improved = True
                best_gen_loss = loss

        loss_history.append(best_loss)

        # Imprimir progreso cada 5 generaciones o si hubo mejora
        if verbose and (gen % 5 == 0 or improved):
            status = "✨ MEJORA" if improved else "→ igual"
            print(
                f"[Evolución] Gen {gen:3d}/{generations} - Loss: {best_gen_loss:.6f} {status}"
            )

    if verbose:
        print(f"\n🎯 Evolución completada - Loss final: {best_loss:.6f}")
        improvement = ((loss_history[0] - best_loss) / loss_history[0]) * 100
        print(f"   Mejora total: {improvement:.2f}%")

    return parent, loss_history


def cross_evaluate_reasoners(
    graph: Any,
    reasoners: List[Any],
    X: np.ndarray,
    Y: np.ndarray,
    mode: str = "softmax",
) -> Dict[int, float]:
    """
    Evalúa múltiples reasoners en el mismo dataset para comparación.
    
    Útil para comparar diferentes configuraciones o estados del Reasoner.
    
    Args:
        graph: Instancia de CognitiveGraphHybrid
        reasoners: Lista de Reasoner a evaluar
        X: Array de inputs (N, n_features)
        Y: Array de targets (N, n_outputs)
        mode: Modo de gating
        
    Returns:
        Diccionario {índice: loss} para cada reasoner
    """
    results = {}

    for idx, reasoner in enumerate(reasoners):
        loss = evaluate_reasoner(graph, reasoner, X, Y, mode=mode)
        results[idx] = loss

    return results


def extract_gates_history(
    graph: Any, reasoner: Any, X: np.ndarray, mode: str = "softmax"
) -> List[Dict[str, float]]:
    """
    Extrae el historial de gates aplicados por el reasoner en un dataset.
    
    Útil para visualización y análisis de las decisiones del reasoner.
    
    Args:
        graph: Instancia de CognitiveGraphHybrid
        reasoner: Reasoner a analizar
        X: Array de inputs (N, n_features)
        mode: Modo de gating
        
    Returns:
        Lista de diccionarios {nombre_bloque: gate_value} para cada input
    """
    history = []

    for x in X:
        # Forward para actualizar planes
        _ = graph.forward({"sensor": x.tolist()})

        # Forward con reasoner
        _ = graph.forward_with_reasoner({"sensor": x.tolist()}, reasoner, mode=mode)

        # Extraer gates aplicados
        if hasattr(graph, "last_gates"):
            gates = dict(graph.last_gates)  # type: ignore[attr-defined]
            history.append(gates)
        else:
            # Fallback si no hay gates registrados
            history.append({name: 1.0 for name in graph.blocks.keys()})

    return history
