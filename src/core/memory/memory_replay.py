from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

import numpy as np

try:
    from src.core.memory.episodic_memory import EpisodicMemory  # type: ignore
except ModuleNotFoundError:
    from core.memory.episodic_memory import EpisodicMemory  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    try:
        from src.core.memory.consolidation import MemoryConsolidation  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        from core.memory.consolidation import MemoryConsolidation  # type: ignore


class MemoryReplaySystem:
    """Sistema central de memoria episódica y consolidación."""

    def __init__(self, graph: Any, monitor: Any, capacity: int = 512) -> None:
        self.graph = graph
        self.monitor = monitor
        self.memory = EpisodicMemory(capacity=capacity)
        self.consolidation = self._build_consolidation(graph)
        self.sleep_cycles = 0
        setattr(self.graph, "memory_system", self)

    def record_experience(
        self,
        inputs: Any,
        target: Any,
        outputs: Dict[str, Any],
        loss: float,
    ) -> None:
        attention_snapshot: Dict[str, Dict[str, np.ndarray]] = {}
        last_attention = getattr(self.graph, "last_attention", {}) or {}
        for dest, src_map in last_attention.items():
            attention_snapshot[dest] = {
                src: np.array(weights, dtype=np.float32).copy() for src, weights in src_map.items()
            }

        self.memory.store(
            deepcopy(inputs),
            deepcopy(target),
            {k: np.array(v, dtype=np.float32).copy() for k, v in outputs.items()},
            float(loss),
            attention_snapshot,
        )

    def sleep_and_replay(self) -> float:
        """Inicia una fase de consolidación con las mejores experiencias."""
        avg_loss = float(self.consolidation.replay_phase())
        self.sleep_cycles += 1
        if self.monitor is not None:
            self.monitor.logger.log(
                "INFO",
                f"🧘 Sleep cycle {self.sleep_cycles} consolidado (avg_loss={avg_loss:.6f})",
            )
        return avg_loss

    def _build_consolidation(self, graph: Any):
        if TYPE_CHECKING:  # pragma: no cover
            consolidation_cls = MemoryConsolidation  # type: ignore[name-defined]
        else:
            try:
                from src.core.memory.consolidation import MemoryConsolidation as consolidation_cls  # type: ignore
            except ModuleNotFoundError:
                from core.memory.consolidation import MemoryConsolidation as consolidation_cls  # type: ignore
        return consolidation_cls(graph, self.memory)
