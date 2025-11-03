from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from src.core.monitor.logger import CognitiveLogger  # type: ignore
except ModuleNotFoundError:
    from core.monitor.logger import CognitiveLogger  # type: ignore

try:
    from src.core.monitor.global_state import record_monitor_state  # type: ignore
except ModuleNotFoundError:
    from core.monitor.global_state import record_monitor_state  # type: ignore


class CognitiveMonitor:
    """Sistema de monitoreo para grafo cognitivo híbrido."""

    def __init__(self, logger: CognitiveLogger | None = None) -> None:
        self.logger = logger or CognitiveLogger(to_file=False)
        self.activations: Dict[str, np.ndarray] = {}
        self.attention_weights: Dict[str, Dict[str, np.ndarray]] = {}
        self.loss_history: List[float] = []

    # ----------------------- Tracking primitives -----------------------
    def track_activations(self, block_name: str, output: np.ndarray) -> None:
        self.activations[block_name] = output
        mean_val = float(np.mean(output))
        self.logger.log("INFO", f"Activación {block_name} = {mean_val:.4f}")
        self._sync_state()

    def track_attention(self, src: str, dest: str, weights: np.ndarray) -> None:
        dest_map = self.attention_weights.setdefault(dest, {})
        dest_map[src] = weights
        avg_weight = float(np.mean(weights))
        self.logger.log("DEBUG", f"Atención {src}->{dest} = {avg_weight:.3f}")
        self._sync_state()

    def track_loss(self, loss_value: float) -> None:
        self.loss_history.append(loss_value)
        self.logger.log("INFO", f"Pérdida global = {loss_value:.6f}")
        self._sync_state()

    # ----------------------- Reporting utilities -----------------------
    def summary(self) -> None:
        self.logger.log("INFO", "--- RESUMEN COGNITIVO ---")
        for name, act in self.activations.items():
            self.logger.log("INFO", f"Bloque {name}: media activación {float(np.mean(act)):.4f}")
        if self.loss_history:
            self.logger.log("INFO", f"Última pérdida: {self.loss_history[-1]:.6f}")
        for dest, sources in self.attention_weights.items():
            for src, weights in sources.items():
                self.logger.log(
                    "INFO",
                    f"Atención {src}->{dest}: media {float(np.mean(weights)):.4f}",
                )

    def _sync_state(self) -> None:
        payload = {
            "loss_history": list(self.loss_history),
            "activations": {
                name: np.asarray(value).tolist() for name, value in self.activations.items()
            },
            "attention": {
                dest: {src: np.asarray(weights).tolist() for src, weights in sources.items()}
                for dest, sources in self.attention_weights.items()
            },
        }
        record_monitor_state(payload)
