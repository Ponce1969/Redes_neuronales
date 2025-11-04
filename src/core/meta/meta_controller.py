"""Controlador de meta-aprendizaje para ajustar hiperparámetros en vivo."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.meta.rules import adaptive_focus, adaptive_lr, adaptive_sleep


class MetaLearningController:
    """Observa métricas en tiempo real y ajusta hiperparámetros clave."""

    def __init__(self, trainer: Any, memory_system: Any, monitor: Any) -> None:
        self.trainer = trainer
        self.memory_system = memory_system
        self.monitor = monitor

        self.lr = getattr(trainer.optimizer, "lr", 0.01)
        self.focus = 1.0
        self.sleep_interval = 3
        self.prev_loss = float("inf")

    # ------------------------------------------------------------------
    # Bucle de observación y ajuste
    # ------------------------------------------------------------------
    def observe_and_adjust(self, epoch: int, curr_loss: float) -> None:
        """Aplica reglas adaptativas sobre lr, atención y sueño."""

        # Learning rate dinámico
        new_lr = adaptive_lr(self.prev_loss, curr_loss, self.lr)
        if abs(new_lr - self.lr) > 1e-8:
            setattr(self.trainer.optimizer, "lr", new_lr)
            self.monitor.logger.log("META", f"LR ajustado → {new_lr:.5f}")
            self.lr = new_lr

        # Foco atencional
        if getattr(self.monitor, "attention_weights", None):
            mean_att = self._mean_attention()
            new_focus = adaptive_focus(mean_att, self.focus)
            if abs(new_focus - self.focus) > 1e-8:
                self.focus = new_focus
                self.monitor.logger.log("META", f"Focus atencional → {new_focus:.3f}")

        # Intervalo de consolidación
        loss_history = getattr(self.monitor, "loss_history", [])
        self.sleep_interval = adaptive_sleep(loss_history, self.sleep_interval)
        self.monitor.logger.log("META", f"Próximo sleep cada {self.sleep_interval} épocas")

        self.prev_loss = curr_loss

    def maybe_sleep(self, epoch: int) -> None:
        """Ejecuta consolidación adaptativa si corresponde."""

        if epoch > 0 and epoch % self.sleep_interval == 0:
            self.monitor.logger.log("META", "Entrando en fase de sueño adaptativa")
            self.memory_system.sleep_and_replay()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mean_attention(self) -> float:
        weights = []
        for dest in self.monitor.attention_weights.values():
            for src_weights in dest.values():
                weights.append(float(np.mean(src_weights)))
        return float(np.mean(weights)) if weights else 0.0
