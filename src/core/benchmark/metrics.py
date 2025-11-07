"""
Métricas Científicas de Benchmark - Análisis estadístico completo.

Define métricas avanzadas para evaluar y comparar configuraciones.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class BenchmarkMetrics:
    """
    Métricas científicas completas para un run de benchmark.
    
    Incluye:
    - Performance (loss, accuracy)
    - Convergencia (época, tiempo)
    - Estabilidad (std, variance)
    - Gates cognitivos (diversity, entropy, consistency)
    - Eficiencia (epochs/sec, tiempo total)
    - Generalización (train/test gap)
    """
    
    # ========================================================================
    # Performance Final
    # ========================================================================
    final_loss: float
    final_accuracy: float
    best_loss: float
    best_accuracy: float
    
    # ========================================================================
    # Convergencia
    # ========================================================================
    convergence_epoch: Optional[int] = None  # Primera época bajo threshold
    time_to_threshold: Optional[float] = None  # Segundos hasta threshold
    converged: bool = False
    
    # ========================================================================
    # Estabilidad durante el Training
    # ========================================================================
    loss_std: float = 0.0  # Desviación estándar del loss
    loss_variance: float = 0.0
    training_stability: float = 1.0  # 1 - (std / mean), más alto = más estable
    loss_trend_slope: float = 0.0  # Pendiente de regresión lineal del loss
    
    # ========================================================================
    # Gates Cognitivos
    # ========================================================================
    gate_diversity: float = 0.0  # Uniformidad en uso de bloques
    gate_entropy: float = 0.0  # Entropía de Shannon
    gate_consistency: float = 0.0  # Consistencia entre epochs
    gate_utilization: float = 0.0  # % de bloques activos (>0.1)
    dominant_gates: List[int] = field(default_factory=list)  # Indices más usados
    
    # ========================================================================
    # Eficiencia
    # ========================================================================
    total_epochs: int = 0
    total_training_time: float = 0.0  # Segundos
    epochs_per_second: float = 0.0
    samples_per_second: float = 0.0
    
    # ========================================================================
    # Generalización
    # ========================================================================
    train_loss: float = 0.0
    test_loss: float = 0.0
    generalization_gap: float = 0.0  # test_loss - train_loss
    overfitting_score: float = 0.0  # max(0, gap / train_loss)
    
    # ========================================================================
    # Curriculum-Specific (si aplica)
    # ========================================================================
    stages_completed: Optional[int] = None
    avg_epochs_per_stage: Optional[float] = None
    stage_progression_rate: Optional[float] = None  # stages/epoch
    curriculum_efficiency: Optional[float] = None  # loss_improvement / epochs
    
    # ========================================================================
    # Meta-información
    # ========================================================================
    run_id: str = ""
    config_hash: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            # Performance
            "final_loss": float(self.final_loss),
            "final_accuracy": float(self.final_accuracy),
            "best_loss": float(self.best_loss),
            "best_accuracy": float(self.best_accuracy),
            
            # Convergencia
            "convergence_epoch": self.convergence_epoch,
            "time_to_threshold": self.time_to_threshold,
            "converged": self.converged,
            
            # Estabilidad
            "loss_std": float(self.loss_std),
            "loss_variance": float(self.loss_variance),
            "training_stability": float(self.training_stability),
            "loss_trend_slope": float(self.loss_trend_slope),
            
            # Gates
            "gate_diversity": float(self.gate_diversity),
            "gate_entropy": float(self.gate_entropy),
            "gate_consistency": float(self.gate_consistency),
            "gate_utilization": float(self.gate_utilization),
            "dominant_gates": self.dominant_gates,
            
            # Eficiencia
            "total_epochs": self.total_epochs,
            "total_training_time": float(self.total_training_time),
            "epochs_per_second": float(self.epochs_per_second),
            "samples_per_second": float(self.samples_per_second),
            
            # Generalización
            "train_loss": float(self.train_loss),
            "test_loss": float(self.test_loss),
            "generalization_gap": float(self.generalization_gap),
            "overfitting_score": float(self.overfitting_score),
            
            # Curriculum
            "stages_completed": self.stages_completed,
            "avg_epochs_per_stage": self.avg_epochs_per_stage,
            "stage_progression_rate": self.stage_progression_rate,
            "curriculum_efficiency": self.curriculum_efficiency,
            
            # Meta
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkMetrics':
        """Crea instancia desde diccionario."""
        return cls(**data)
    
    @classmethod
    def aggregate(
        cls,
        metrics_list: List['BenchmarkMetrics'],
        confidence_level: float = 0.95,
    ) -> 'AggregatedMetrics':
        """
        Agrega múltiples runs calculando estadísticas.
        
        Args:
            metrics_list: Lista de métricas de diferentes runs
            confidence_level: Nivel de confianza para intervalos
        
        Returns:
            AggregatedMetrics con mean, std, CI
        """
        if not metrics_list:
            raise ValueError("metrics_list no puede estar vacía")
        
        n_runs = len(metrics_list)
        
        # Agregar cada métrica numérica
        aggregated = {}
        
        numeric_fields = [
            "final_loss", "final_accuracy", "best_loss", "best_accuracy",
            "loss_std", "loss_variance", "training_stability", "loss_trend_slope",
            "gate_diversity", "gate_entropy", "gate_consistency", "gate_utilization",
            "total_epochs", "total_training_time", "epochs_per_second",
            "train_loss", "test_loss", "generalization_gap", "overfitting_score",
        ]
        
        for field_name in numeric_fields:
            values = [getattr(m, field_name) for m in metrics_list]
            
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n_runs > 1 else 0.0
            
            # Intervalo de confianza
            if n_runs > 1:
                ci_low, ci_high = stats.t.interval(
                    confidence_level,
                    df=n_runs - 1,
                    loc=mean,
                    scale=stats.sem(values)
                )
            else:
                ci_low = ci_high = mean
            
            aggregated[field_name] = {
                "mean": float(mean),
                "std": float(std),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        
        # Agregados especiales
        convergence_epochs = [
            m.convergence_epoch for m in metrics_list
            if m.convergence_epoch is not None
        ]
        
        if convergence_epochs:
            aggregated["convergence_epoch"] = {
                "mean": float(np.mean(convergence_epochs)),
                "std": float(np.std(convergence_epochs, ddof=1)) if len(convergence_epochs) > 1 else 0.0,
            }
        
        return AggregatedMetrics(
            n_runs=n_runs,
            confidence_level=confidence_level,
            aggregated=aggregated,
            config_hash=metrics_list[0].config_hash,
        )


@dataclass
class AggregatedMetrics:
    """
    Métricas agregadas de múltiples runs.
    
    Incluye mean, std, confidence intervals para cada métrica.
    """
    n_runs: int
    confidence_level: float
    aggregated: Dict[str, Dict[str, float]]
    config_hash: str = ""
    
    def get_mean(self, metric: str) -> float:
        """Obtiene la media de una métrica."""
        return self.aggregated[metric]["mean"]
    
    def get_std(self, metric: str) -> float:
        """Obtiene la desviación estándar de una métrica."""
        return self.aggregated[metric]["std"]
    
    def get_ci(self, metric: str) -> Tuple[float, float]:
        """Obtiene el intervalo de confianza de una métrica."""
        return (
            self.aggregated[metric]["ci_low"],
            self.aggregated[metric]["ci_high"]
        )
    
    def format_metric(self, metric: str, precision: int = 4) -> str:
        """
        Formatea una métrica como: mean ± std [CI_low, CI_high]
        
        Args:
            metric: Nombre de la métrica
            precision: Decimales
        
        Returns:
            String formateado
        """
        stats = self.aggregated[metric]
        return (
            f"{stats['mean']:.{precision}f} ± {stats['std']:.{precision}f} "
            f"[{stats['ci_low']:.{precision}f}, {stats['ci_high']:.{precision}f}]"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "n_runs": self.n_runs,
            "confidence_level": self.confidence_level,
            "config_hash": self.config_hash,
            "metrics": self.aggregated,
        }


# ============================================================================
# Funciones Helper para Cálculo de Métricas
# ============================================================================

def calculate_loss_stability(loss_history: List[float]) -> Tuple[float, float, float]:
    """
    Calcula métricas de estabilidad del loss.
    
    Args:
        loss_history: Historial de loss por época
    
    Returns:
        (std, variance, stability_score)
    """
    if not loss_history:
        return 0.0, 0.0, 1.0
    
    loss_array = np.array(loss_history)
    
    std = float(np.std(loss_array))
    variance = float(np.var(loss_array))
    
    mean = np.mean(loss_array)
    stability = 1.0 - (std / mean if mean > 0 else 0.0)
    stability = max(0.0, min(1.0, stability))  # Clip [0, 1]
    
    return std, variance, stability


def calculate_loss_trend(loss_history: List[float]) -> float:
    """
    Calcula la tendencia del loss (pendiente de regresión lineal).
    
    Negativo = mejorando, Positivo = empeorando, ~0 = estable
    
    Args:
        loss_history: Historial de loss por época
    
    Returns:
        Pendiente de la regresión lineal
    """
    if len(loss_history) < 2:
        return 0.0
    
    x = np.arange(len(loss_history))
    y = np.array(loss_history)
    
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def calculate_gate_consistency(gates_history: List[np.ndarray]) -> float:
    """
    Calcula consistencia de gates entre epochs.
    
    Mide qué tan similares son los gates a lo largo del tiempo.
    
    Args:
        gates_history: Lista de arrays de gates por época
    
    Returns:
        Score de consistencia [0, 1] (1 = muy consistente)
    """
    if len(gates_history) < 2:
        return 1.0
    
    # Calcular correlación promedio entre epochs consecutivas
    correlations = []
    
    for i in range(len(gates_history) - 1):
        gates_a = gates_history[i]
        gates_b = gates_history[i + 1]
        
        # Correlación de Pearson
        if len(gates_a) > 1:
            corr, _ = stats.pearsonr(gates_a, gates_b)
            correlations.append(corr)
    
    if not correlations:
        return 1.0
    
    avg_corr = np.mean(correlations)
    
    # Convertir de [-1, 1] a [0, 1]
    consistency = (avg_corr + 1) / 2
    
    return float(consistency)


def find_convergence_epoch(
    loss_history: List[float],
    threshold: float = 0.05,
    patience: int = 5,
) -> Optional[int]:
    """
    Encuentra la época donde converge (loss bajo threshold por `patience` epochs).
    
    Args:
        loss_history: Historial de loss
        threshold: Threshold de convergencia
        patience: Cuántas epochs consecutivas bajo threshold
    
    Returns:
        Época de convergencia o None
    """
    if len(loss_history) < patience:
        return None
    
    consecutive = 0
    
    for epoch, loss in enumerate(loss_history):
        if loss <= threshold:
            consecutive += 1
            if consecutive >= patience:
                return epoch - patience + 1
        else:
            consecutive = 0
    
    return None


def calculate_dominant_gates(
    gates_history: List[np.ndarray],
    top_n: int = 3,
) -> List[int]:
    """
    Identifica los gates más usados.
    
    Args:
        gates_history: Historial de gates
        top_n: Cuántos top gates retornar
    
    Returns:
        Indices de los gates dominantes
    """
    if not gates_history:
        return []
    
    # Promediar uso de cada gate
    avg_gates = np.mean(gates_history, axis=0)
    
    # Top N indices
    top_indices = np.argsort(avg_gates)[-top_n:][::-1]
    
    return top_indices.tolist()
