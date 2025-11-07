"""
API Router para control del Reasoner.

Endpoints para gestionar el Reasoner: consultar estado, obtener gates,
evolucionar en background, persistir configuraciones, etc.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import CognitiveAppState, get_app_state

router = APIRouter(prefix="/reasoner", tags=["Reasoner"])


# ========================================================================
# MODELOS PYDANTIC
# ========================================================================


class GatingRequest(BaseModel):
    """Request para calcular gates sobre vectores latentes."""

    z_vectors: List[List[float]] = Field(
        ..., description="Lista de vectores latentes (uno por bloque)"
    )
    mode: str = Field(default="softmax", description="Modo: softmax, topk, threshold")
    temp: float = Field(default=1.0, description="Temperatura para softmax")
    top_k: int = Field(default=2, description="Número de bloques en modo topk")


class EvolutionRequest(BaseModel):
    """Request para iniciar evolución del Reasoner."""

    generations: int = Field(default=50, ge=1, le=500, description="Generaciones a evolucionar")
    pop_size: int = Field(default=8, ge=2, le=20, description="Tamaño de población")
    mutation_scale: float = Field(default=0.03, ge=0.001, le=0.1, description="Escala de mutación")


# ========================================================================
# ENDPOINTS
# ========================================================================


@router.get("/status")
async def get_reasoner_status(
    state: CognitiveAppState = Depends(get_app_state),
) -> Dict:
    """
    Obtiene el estado actual del Reasoner.
    
    Returns:
        Diccionario con métricas: running, generation, best_loss, etc.
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    return state.reasoner_manager.status()


@router.get("/gates")
async def get_recent_gates(
    n: int = 10, state: CognitiveAppState = Depends(get_app_state)
) -> Dict:
    """
    Obtiene los últimos N gates calculados.
    
    Args:
        n: Número de gates recientes (default: 10, max: 100)
        
    Returns:
        Lista de diccionarios {block_index: gate_value}
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    n = min(n, 100)  # Limitar a 100
    gates_history = state.reasoner_manager.get_recent_gates(n)

    return {
        "gates_history": gates_history,
        "count": len(gates_history),
    }


@router.post("/predict")
async def predict_with_reasoner(
    request: GatingRequest, state: CognitiveAppState = Depends(get_app_state)
) -> Dict:
    """
    Calcula gates para un conjunto de vectores latentes.
    
    Args:
        request: GatingRequest con z_vectors, mode, etc.
        
    Returns:
        Diccionario con gates calculados
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    try:
        # Convertir a arrays NumPy
        z_per_block = [
            np.array(z, dtype=np.float32).reshape(1, -1) for z in request.z_vectors
        ]

        # Calcular gates
        gates = state.reasoner_manager.decide(
            z_per_block,
            mode=request.mode,
            temp=request.temp,
            top_k=request.top_k,
        )

        return {"gates": gates, "mode": request.mode}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predict: {str(e)}")


@router.post("/evolve")
async def evolve_reasoner(
    request: EvolutionRequest,
    background_tasks: BackgroundTasks,
    state: CognitiveAppState = Depends(get_app_state),
) -> Dict:
    """
    Inicia evolución del Reasoner en background.
    
    Args:
        request: EvolutionRequest con generations, pop_size, etc.
        background_tasks: FastAPI background tasks
        
    Returns:
        Confirmación del inicio de evolución
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    if state.reasoner_manager.running:
        raise HTTPException(status_code=409, detail="Evolution already running")

    # Función de evaluación REAL usando el grafo
    def evaluate_on_graph(reasoner) -> float:
        """Evalúa el reasoner en XOR usando el grafo del primer agente."""
        try:
            graph = state.society.agents[0].graph

            # Dataset XOR (puedes parametrizar esto en el futuro)
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
            Y = np.array([0, 1, 1, 0], dtype=np.float32)

            errors = []
            for x, y_true in zip(X, Y):
                # Forward normal para computar z_plan
                _ = graph.forward({"input": x.tolist()})

                # Forward con reasoner
                out_map = graph.forward_with_reasoner(
                    {"input": x.tolist()}, reasoner, mode="softmax"
                )

                # Obtener predicción del último bloque
                y_pred = list(out_map.values())[-1].data.squeeze()
                error = float((y_pred - y_true) ** 2)
                errors.append(error)

            return float(np.mean(errors))

        except Exception as e:
            print(f"[Reasoner Evolution] Error en evaluación: {e}")
            return 1.0  # Pérdida alta por defecto

    # Lanzar evolución
    started = state.reasoner_manager.evolve_async(
        evaluate_fn=evaluate_on_graph,
        generations=request.generations,
        pop_size=request.pop_size,
        mutation_scale=request.mutation_scale,
    )

    if not started:
        raise HTTPException(status_code=409, detail="Failed to start evolution")

    return {
        "started": True,
        "generations": request.generations,
        "pop_size": request.pop_size,
        "mutation_scale": request.mutation_scale,
    }


@router.post("/evolve/stop")
async def stop_evolution(state: CognitiveAppState = Depends(get_app_state)) -> Dict:
    """
    Detiene la evolución en curso.
    
    Returns:
        Confirmación de detención
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    was_running = state.reasoner_manager.stop_evolution()

    return {"stopped": was_running, "message": "Evolution stopped" if was_running else "No evolution running"}


@router.post("/save")
async def save_reasoner(state: CognitiveAppState = Depends(get_app_state)) -> Dict:
    """
    Guarda el estado actual del Reasoner.
    
    Returns:
        Confirmación de guardado
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    success = state.reasoner_manager.save("data/persistence/reasoner_state")

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save reasoner")

    return {"saved": True, "path": "data/persistence/reasoner_state.npz"}


@router.post("/load")
async def load_reasoner(state: CognitiveAppState = Depends(get_app_state)) -> Dict:
    """
    Carga el estado del Reasoner desde disco.
    
    Returns:
        Confirmación de carga
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    success = state.reasoner_manager.load("data/persistence/reasoner_state")

    if not success:
        raise HTTPException(status_code=404, detail="Reasoner state file not found")

    return {"loaded": True, "path": "data/persistence/reasoner_state.npz"}


@router.post("/reset")
async def reset_reasoner_stats(state: CognitiveAppState = Depends(get_app_state)) -> Dict:
    """
    Reinicia estadísticas del Reasoner (útil para testing).
    
    Returns:
        Confirmación de reset
    """
    if not state.reasoner_manager:
        raise HTTPException(status_code=503, detail="ReasonerManager not initialized")

    state.reasoner_manager.reset_stats()

    return {"reset": True, "message": "Statistics reset successfully"}
