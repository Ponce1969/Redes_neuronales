"""
API Router para Curriculum Learning.

Endpoints para controlar el sistema de curriculum learning desde HTTP.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from api.dependencies import get_app_state
from core.curriculum import (
    CurriculumManager,
    CurriculumStage,
    create_standard_curriculum,
    tasks,
)


router = APIRouter(prefix="/curriculum", tags=["Curriculum Learning"])


# ============================================================================
# Pydantic Models
# ============================================================================

class StageConfig(BaseModel):
    """Configuración de una etapa del curriculum."""
    name: str = Field(..., description="Nombre de la tarea (identity, xor, parity, etc.)")
    difficulty: int = Field(1, ge=1, le=10, description="Nivel de dificultad (1-10)")
    max_epochs: int = Field(100, ge=1, description="Máximo número de épocas")
    success_threshold: float = Field(0.05, gt=0, description="Threshold de éxito (loss)")
    fail_threshold: float = Field(0.5, gt=0, description="Threshold de fallo (loss)")
    log_interval: int = Field(10, ge=1, description="Cada cuántas épocas loggear")
    evolution_generations: int = Field(1, ge=1, description="Generaciones de evolución por época")
    mutation_scale: float = Field(0.03, gt=0, description="Escala de mutación")


class StartCurriculumRequest(BaseModel):
    """Request para iniciar curriculum."""
    stages: Optional[List[StageConfig]] = Field(
        None,
        description="Etapas personalizadas (si None, usa curriculum estándar)"
    )
    auto_save: bool = Field(True, description="Auto-save después de cada etapa")
    checkpoint_dir: str = Field("data/curriculum", description="Directorio de checkpoints")


class CurriculumStatusResponse(BaseModel):
    """Response con el estado del curriculum."""
    running: bool
    paused: bool
    current_stage_idx: int
    total_stages: int
    current_stage_name: Optional[str]
    stages_completed: int
    progress: float
    history: List[Dict[str, Any]]
    stage_names: List[str]


# ============================================================================
# Global curriculum manager (inicializado bajo demanda)
# ============================================================================

_curriculum_manager: Optional[CurriculumManager] = None


def get_curriculum_manager(state = Depends(get_app_state)) -> CurriculumManager:
    """
    Obtiene o crea el CurriculumManager global.
    
    Args:
        state: CognitiveAppState desde dependencia
    
    Returns:
        Instancia de CurriculumManager
    """
    global _curriculum_manager
    
    if _curriculum_manager is None:
        if state.reasoner_manager is None:
            raise HTTPException(
                status_code=500,
                detail="ReasonerManager no inicializado en el servidor"
            )
        
        if state.graph is None:
            raise HTTPException(
                status_code=500,
                detail="CognitiveGraph no inicializado en el servidor"
            )
        
        _curriculum_manager = CurriculumManager(
            reasoner_manager=state.reasoner_manager,
            graph=state.graph,
        )
    
    return _curriculum_manager


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=Dict[str, Any])
async def start_curriculum(
    request: StartCurriculumRequest,
    background_tasks: BackgroundTasks,
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Inicia el curriculum learning en background.
    
    Args:
        request: Configuración del curriculum
        background_tasks: FastAPI background tasks
        manager: CurriculumManager inyectado
    
    Returns:
        Confirmación con lista de etapas
    """
    if manager.running:
        raise HTTPException(status_code=400, detail="El curriculum ya está corriendo")
    
    # Resetear manager
    manager.reset()
    
    # Añadir etapas
    if request.stages:
        # Etapas personalizadas
        for stage_config in request.stages:
            try:
                task_func = tasks.get_task(stage_config.name)
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tarea '{stage_config.name}' no encontrada. "
                    f"Disponibles: {list(tasks.TASK_REGISTRY.keys())}"
                )
            
            stage = CurriculumStage(
                name=stage_config.name,
                task_generator=task_func,
                difficulty=stage_config.difficulty,
                max_epochs=stage_config.max_epochs,
                success_threshold=stage_config.success_threshold,
                fail_threshold=stage_config.fail_threshold,
                log_interval=stage_config.log_interval,
                evolution_generations=stage_config.evolution_generations,
                mutation_scale=stage_config.mutation_scale,
            )
            manager.add_stage(stage)
    else:
        # Curriculum estándar
        for stage in create_standard_curriculum():
            manager.add_stage(stage)
    
    # Lanzar en background
    background_tasks.add_task(manager.run)
    
    return {
        "started": True,
        "total_stages": len(manager.stages),
        "stage_names": [s.name for s in manager.stages],
        "auto_save": request.auto_save,
    }


@router.get("/status", response_model=CurriculumStatusResponse)
async def get_curriculum_status(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Obtiene el estado actual del curriculum.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Estado completo del curriculum
    """
    status = manager.status()
    return CurriculumStatusResponse(**status)


@router.post("/pause")
async def pause_curriculum(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Pausa la ejecución del curriculum.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Confirmación
    """
    if not manager.running:
        raise HTTPException(status_code=400, detail="El curriculum no está corriendo")
    
    manager.pause()
    return {"paused": True}


@router.post("/resume")
async def resume_curriculum(
    background_tasks: BackgroundTasks,
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Reanuda la ejecución del curriculum.
    
    Args:
        background_tasks: FastAPI background tasks
        manager: CurriculumManager inyectado
    
    Returns:
        Confirmación
    """
    if manager.running:
        raise HTTPException(status_code=400, detail="El curriculum ya está corriendo")
    
    manager.resume()
    background_tasks.add_task(manager.run)
    
    return {"resumed": True}


@router.post("/reset")
async def reset_curriculum(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Resetea el curriculum para empezar desde cero.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Confirmación
    """
    manager.reset()
    return {"reset": True}


@router.get("/history")
async def get_curriculum_history(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Obtiene el historial completo de etapas completadas.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Lista de registros históricos
    """
    return {"history": manager.history}


@router.get("/checkpoints")
async def list_checkpoints(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Lista todos los checkpoints disponibles.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Lista de checkpoints con metadata
    """
    backups = manager.checkpointer.get_backup_list()
    return {"checkpoints": backups}


@router.post("/export")
async def export_curriculum_results(
    manager: CurriculumManager = Depends(get_curriculum_manager),
):
    """
    Exporta los resultados del curriculum en formato JSON.
    
    Args:
        manager: CurriculumManager inyectado
    
    Returns:
        Resultados completos del curriculum
    """
    status = manager.status()
    
    return {
        "curriculum_export": {
            "status": status,
            "summary": {
                "total_epochs": sum(r['epochs'] for r in manager.history) if manager.history else 0,
                "avg_loss": (
                    sum(r['mse_loss'] for r in manager.history) / len(manager.history)
                    if manager.history else 0
                ),
                "completion_rate": len(manager.history) / len(manager.stages) if manager.stages else 0,
            },
        }
    }
