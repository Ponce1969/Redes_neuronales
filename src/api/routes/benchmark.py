"""
API Router para Benchmarks.

Endpoints para ejecutar y gestionar benchmarks desde HTTP.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from api.dependencies import get_app_state
from core.benchmark import (
    BenchmarkSuite,
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    list_configs,
    ReportGenerator,
)


router = APIRouter(prefix="/benchmark", tags=["Benchmarking"])


# ============================================================================
# Pydantic Models
# ============================================================================

class RunBenchmarkRequest(BaseModel):
    """Request para ejecutar un benchmark."""
    config_name: str = Field(..., description="Nombre de la configuración")
    save_results: bool = Field(True, description="Guardar resultados")


class RunComparisonRequest(BaseModel):
    """Request para ejecutar comparación."""
    config_names: List[str] = Field(..., description="Lista de configs a comparar")
    metric: str = Field("final_loss", description="Métrica principal")


class BenchmarkStatusResponse(BaseModel):
    """Response con estado del benchmark."""
    running: bool
    current_config: Optional[str]
    progress: float
    results_count: int


# ============================================================================
# Global state
# ============================================================================

_benchmark_suite: Optional[BenchmarkSuite] = None
_benchmark_running: bool = False
_current_config: Optional[str] = None


def get_benchmark_suite() -> BenchmarkSuite:
    """Obtiene o crea BenchmarkSuite global."""
    global _benchmark_suite
    
    if _benchmark_suite is None:
        _benchmark_suite = BenchmarkSuite(
            output_dir="data/benchmarks/results",
            verbose=True,
        )
    
    return _benchmark_suite


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/configs", response_model=List[str])
async def get_available_configs():
    """
    Lista todas las configuraciones disponibles.
    
    Returns:
        Lista de nombres de configuraciones
    """
    return list_configs()


@router.get("/config/{config_name}")
async def get_config_details(config_name: str):
    """
    Obtiene detalles de una configuración.
    
    Args:
        config_name: Nombre de la configuración
    
    Returns:
        Detalles completos de la configuración
    """
    if config_name not in BENCHMARK_CONFIGS:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' no encontrada. Disponibles: {list_configs()}"
        )
    
    config = BENCHMARK_CONFIGS[config_name]
    return config.to_dict()


@router.post("/run")
async def run_benchmark(
    request: RunBenchmarkRequest,
    background_tasks: BackgroundTasks,
    state=Depends(get_app_state),
):
    """
    Ejecuta un benchmark.
    
    Args:
        request: Configuración del benchmark
        background_tasks: FastAPI background tasks
        state: CognitiveAppState
    
    Returns:
        Confirmación con run ID
    """
    global _benchmark_running, _current_config
    
    if _benchmark_running:
        raise HTTPException(status_code=400, detail="Ya hay un benchmark corriendo")
    
    # Validar config
    if request.config_name not in BENCHMARK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Config '{request.config_name}' no encontrada"
        )
    
    # Validar state
    if state.reasoner_manager is None or state.graph is None:
        raise HTTPException(
            status_code=500,
            detail="ReasonerManager o Graph no inicializados"
        )
    
    # Obtener config
    config = BENCHMARK_CONFIGS[request.config_name]
    suite = get_benchmark_suite()
    
    # Ejecutar en background
    def run_benchmark_task():
        global _benchmark_running, _current_config
        
        try:
            _benchmark_running = True
            _current_config = request.config_name
            
            result = suite.run_single(
                config=config,
                reasoner_manager=state.reasoner_manager,
                graph=state.graph,
                save_results=request.save_results,
            )
            
            print(f"✅ Benchmark completado: {result.provenance.run_id}")
            
        finally:
            _benchmark_running = False
            _current_config = None
    
    background_tasks.add_task(run_benchmark_task)
    
    return {
        "started": True,
        "config_name": request.config_name,
        "config_hash": config.hash(),
        "n_runs": config.n_runs,
    }


@router.post("/compare")
async def run_comparison(
    request: RunComparisonRequest,
    background_tasks: BackgroundTasks,
    state=Depends(get_app_state),
):
    """
    Ejecuta comparación de múltiples configs.
    
    Args:
        request: Configuración de la comparación
        background_tasks: FastAPI background tasks
        state: CognitiveAppState
    
    Returns:
        Confirmación
    """
    global _benchmark_running
    
    if _benchmark_running:
        raise HTTPException(status_code=400, detail="Ya hay un benchmark corriendo")
    
    # Validar configs
    for config_name in request.config_names:
        if config_name not in BENCHMARK_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Config '{config_name}' no encontrada"
            )
    
    # Validar state
    if state.reasoner_manager is None or state.graph is None:
        raise HTTPException(
            status_code=500,
            detail="ReasonerManager o Graph no inicializados"
        )
    
    # Obtener configs
    configs = [BENCHMARK_CONFIGS[name] for name in request.config_names]
    suite = get_benchmark_suite()
    
    # Ejecutar en background
    def run_comparison_task():
        global _benchmark_running
        
        try:
            _benchmark_running = True
            
            report = suite.run_comparison(
                configs=configs,
                reasoner_manager=state.reasoner_manager,
                graph=state.graph,
                metric=request.metric,
            )
            
            # Guardar reporte
            output_dir = Path("data/benchmarks/reports") / f"comparison_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
            report.save(output_dir)
            
            # Generar reportes multi-formato
            generator = ReportGenerator()
            generator.generate_all(report, output_dir)
            
            print(f"✅ Comparación completada: {output_dir}")
            
        finally:
            _benchmark_running = False
    
    background_tasks.add_task(run_comparison_task)
    
    return {
        "started": True,
        "n_configs": len(request.config_names),
        "config_names": request.config_names,
        "metric": request.metric,
    }


@router.get("/status", response_model=BenchmarkStatusResponse)
async def get_benchmark_status():
    """
    Obtiene estado actual del benchmark.
    
    Returns:
        Estado del benchmark
    """
    # Contar resultados guardados
    results_dir = Path("data/benchmarks/results")
    results_count = len(list(results_dir.glob("*.json"))) if results_dir.exists() else 0
    
    return BenchmarkStatusResponse(
        running=_benchmark_running,
        current_config=_current_config,
        progress=0.0,  # TODO: tracking granular
        results_count=results_count,
    )


@router.get("/results")
async def list_results():
    """
    Lista todos los resultados de benchmarks.
    
    Returns:
        Lista de resultados con metadata
    """
    results_dir = Path("data/benchmarks/results")
    
    if not results_dir.exists():
        return {"results": []}
    
    results = []
    
    for result_file in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            
            results.append({
                "run_id": data["provenance"]["run_id"],
                "config_name": data["config"]["name"],
                "config_hash": data["config"]["config_hash"],
                "timestamp": data["provenance"]["timestamp"],
                "n_runs": data["metrics"]["n_runs"],
                "final_loss_mean": data["metrics"]["metrics"]["final_loss"]["mean"],
            })
        
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    return {"results": results}


@router.get("/result/{run_id}")
async def get_result_detail(run_id: str):
    """
    Obtiene detalles completos de un resultado.
    
    Args:
        run_id: ID del run
    
    Returns:
        Resultado completo
    """
    result_file = Path(f"data/benchmarks/results/{run_id}.json")
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Resultado '{run_id}' no encontrado")
    
    with open(result_file, "r") as f:
        data = json.load(f)
    
    return data


@router.delete("/results")
async def clear_results():
    """
    Limpia todos los resultados guardados.
    
    Returns:
        Confirmación
    """
    results_dir = Path("data/benchmarks/results")
    
    if not results_dir.exists():
        return {"cleared": 0}
    
    count = 0
    for result_file in results_dir.glob("*.json"):
        result_file.unlink()
        count += 1
    
    return {"cleared": count}


@router.get("/reports")
async def list_reports():
    """
    Lista todos los reportes de comparación.
    
    Returns:
        Lista de reportes
    """
    reports_dir = Path("data/benchmarks/reports")
    
    if not reports_dir.exists():
        return {"reports": []}
    
    reports = []
    
    for report_dir in sorted(reports_dir.iterdir(), reverse=True):
        if report_dir.is_dir():
            summary_file = report_dir / "summary.json"
            
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                
                reports.append({
                    "name": report_dir.name,
                    "timestamp": summary["timestamp"],
                    "n_configs": summary["n_configs"],
                    "config_names": summary["config_names"],
                })
    
    return {"reports": reports}
