"""Módulo de razonamiento cognitivo para control selectivo de bloques."""

from __future__ import annotations

try:
    from src.core.reasoning.reasoner import Reasoner  # type: ignore
    from src.core.reasoning.reasoner_manager import ReasonerManager  # type: ignore
    from src.core.reasoning.training import (  # type: ignore
        cross_evaluate_reasoners,
        evaluate_reasoner,
        evolve_reasoner_on_task,
        extract_gates_history,
    )
except ModuleNotFoundError:
    from core.reasoning.reasoner import Reasoner  # type: ignore
    from core.reasoning.reasoner_manager import ReasonerManager  # type: ignore
    from core.reasoning.training import (  # type: ignore
        cross_evaluate_reasoners,
        evaluate_reasoner,
        evolve_reasoner_on_task,
        extract_gates_history,
    )

__all__ = [
    "Reasoner",
    "ReasonerManager",
    "evaluate_reasoner",
    "evolve_reasoner_on_task",
    "cross_evaluate_reasoners",
    "extract_gates_history",
]
