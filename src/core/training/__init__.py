"""Utilities for training CognitiveGraphHybrid models."""

from .losses import mse_loss, l1_loss, binary_cross_entropy
from .optimizers import SGD, Adam

__all__ = ["mse_loss", "l1_loss", "binary_cross_entropy", "SGD", "Adam", "GraphTrainer"]


def __getattr__(name: str):
    if name == "GraphTrainer":
        from .trainer import GraphTrainer  # type: ignore

        return GraphTrainer
    raise AttributeError(f"module 'core.training' has no attribute {name!r}")
