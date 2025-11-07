"""
Curriculum Learning System - Aprendizaje Progresivo para el Reasoner.

Este módulo implementa un sistema completo de curriculum learning que permite
al Reasoner aprender de manera gradual, desde tareas simples hasta complejas.

Componentes principales:
- tasks: Generadores de tareas de diferentes dificultades
- metrics: Métricas avanzadas para evaluar progreso cognitivo
- curriculum_stage: Definición de etapas individuales
- curriculum_manager: Manager principal que coordina el entrenamiento
- evaluator: Evaluador integrado con el grafo cognitivo
- checkpointer: Sistema de checkpoints automáticos

Uso básico:
    from core.curriculum import CurriculumManager, CurriculumStage, tasks
    
    manager = CurriculumManager(reasoner_manager, graph)
    manager.add_stage(CurriculumStage("xor", tasks.xor_task, difficulty=2))
    manager.run()

Fase: 33
Autor: Neural Core Team
"""

from core.curriculum.tasks import (
    identity_task,
    xor_task,
    parity_task,
    sequence_task,
    reasoning_task,
    memory_task,
    counting_task,
    get_task,
    TASK_REGISTRY,
)

from core.curriculum.metrics import CognitiveMetrics

from core.curriculum.curriculum_stage import (
    CurriculumStage,
    create_standard_curriculum,
)

from core.curriculum.curriculum_manager import CurriculumManager

from core.curriculum.evaluator import (
    CurriculumEvaluator,
    evaluate_reasoner_on_task,
)

from core.curriculum.checkpointer import CurriculumCheckpointer


__all__ = [
    # Tasks
    "identity_task",
    "xor_task",
    "parity_task",
    "sequence_task",
    "reasoning_task",
    "memory_task",
    "counting_task",
    "get_task",
    "TASK_REGISTRY",
    # Metrics
    "CognitiveMetrics",
    # Stages
    "CurriculumStage",
    "create_standard_curriculum",
    # Manager
    "CurriculumManager",
    # Evaluator
    "CurriculumEvaluator",
    "evaluate_reasoner_on_task",
    # Checkpointer
    "CurriculumCheckpointer",
]
