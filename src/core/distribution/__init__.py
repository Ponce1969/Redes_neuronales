"""Distribución de experiencias entre sociedades cognitivas."""

from .distributor import CognitiveDistributor
from .receiver import apply_shared_memory

__all__ = ["CognitiveDistributor", "apply_shared_memory"]
