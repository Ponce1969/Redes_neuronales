"""
Definición de etapas individuales del Curriculum Learning.
"""

from typing import Callable, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    """
    Representa una etapa individual en el curriculum de aprendizaje.
    
    Attributes:
        name: Nombre descriptivo de la etapa (ej: "xor", "parity")
        task_generator: Función que genera (X, Y) para entrenamiento
        difficulty: Nivel de dificultad (1-10, subjetivo pero útil para tracking)
        max_epochs: Máximo número de épocas de entrenamiento
        success_threshold: Loss máximo para considerar la etapa completada
        fail_threshold: Loss mínimo para no fallar completamente
        log_interval: Cada cuántas épocas loggear progreso
        evolution_generations: Generaciones de evolución por época
        mutation_scale: Escala de mutación para evolución
    """
    
    name: str
    task_generator: Callable[[], Tuple[np.ndarray, np.ndarray]]
    difficulty: int = 1
    max_epochs: int = 100
    success_threshold: float = 0.05
    fail_threshold: float = 0.5
    log_interval: int = 10
    evolution_generations: int = 1
    mutation_scale: float = 0.03
    
    # Campos calculados automáticamente
    completed: bool = field(default=False, init=False)
    best_metrics: Optional[dict] = field(default=None, init=False)
    epochs_to_complete: Optional[int] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validación de parámetros."""
        if self.difficulty < 1 or self.difficulty > 10:
            raise ValueError(f"Difficulty debe estar entre 1 y 10, recibido: {self.difficulty}")
        
        if self.success_threshold >= self.fail_threshold:
            raise ValueError(
                f"success_threshold ({self.success_threshold}) debe ser < "
                f"fail_threshold ({self.fail_threshold})"
            )
        
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs debe ser >= 1, recibido: {self.max_epochs}")
    
    def mark_completed(self, epochs: int, metrics: dict):
        """
        Marca la etapa como completada con sus métricas finales.
        
        Args:
            epochs: Número de épocas que tomó completarla
            metrics: Métricas finales alcanzadas
        """
        self.completed = True
        self.epochs_to_complete = epochs
        self.best_metrics = metrics.copy()
    
    def reset(self):
        """Resetea el estado de la etapa."""
        self.completed = False
        self.best_metrics = None
        self.epochs_to_complete = None
    
    def to_dict(self) -> dict:
        """Serializa la etapa a diccionario."""
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "max_epochs": self.max_epochs,
            "success_threshold": self.success_threshold,
            "fail_threshold": self.fail_threshold,
            "completed": self.completed,
            "epochs_to_complete": self.epochs_to_complete,
            "best_metrics": self.best_metrics,
        }
    
    def __repr__(self) -> str:
        status = "✅" if self.completed else "⏸️"
        return (
            f"{status} CurriculumStage(name='{self.name}', "
            f"difficulty={self.difficulty}, "
            f"threshold={self.success_threshold})"
        )


def create_standard_curriculum() -> list:
    """
    Crea un curriculum estándar con progresión típica.
    
    Returns:
        Lista de CurriculumStage ordenadas por dificultad
    """
    from core.curriculum.tasks import (
        identity_task,
        xor_task,
        parity_task,
        counting_task,
        sequence_task,
        memory_task,
        reasoning_task,
    )
    
    stages = [
        CurriculumStage(
            name="identity",
            task_generator=lambda: identity_task(n_features=2, samples=16),
            difficulty=1,
            max_epochs=30,
            success_threshold=0.01,
            fail_threshold=0.1,
        ),
        CurriculumStage(
            name="xor",
            task_generator=lambda: xor_task(samples=16),
            difficulty=2,
            max_epochs=50,
            success_threshold=0.02,
            fail_threshold=0.15,
        ),
        CurriculumStage(
            name="parity-3",
            task_generator=lambda: parity_task(n_bits=3, samples=32),
            difficulty=3,
            max_epochs=80,
            success_threshold=0.03,
            fail_threshold=0.2,
        ),
        CurriculumStage(
            name="counting",
            task_generator=lambda: counting_task(max_value=5, samples=32),
            difficulty=4,
            max_epochs=100,
            success_threshold=0.04,
            fail_threshold=0.25,
        ),
        CurriculumStage(
            name="sequence",
            task_generator=lambda: sequence_task(length=4, samples=32),
            difficulty=5,
            max_epochs=150,
            success_threshold=0.05,
            fail_threshold=0.3,
        ),
        CurriculumStage(
            name="memory",
            task_generator=lambda: memory_task(sequence_length=5, samples=32),
            difficulty=6,
            max_epochs=200,
            success_threshold=0.06,
            fail_threshold=0.35,
        ),
        CurriculumStage(
            name="reasoning",
            task_generator=lambda: reasoning_task(samples=32),
            difficulty=7,
            max_epochs=250,
            success_threshold=0.07,
            fail_threshold=0.4,
        ),
    ]
    
    return stages
