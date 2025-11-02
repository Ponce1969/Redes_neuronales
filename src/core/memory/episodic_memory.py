from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np


class EpisodicMemory:
    """Memoria episódica circular para experiencias cognitivas."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=capacity)

    def store(
        self,
        input_data: Any,
        target: Any,
        output: Any,
        loss: float,
        attention_map: Dict[str, Any] | None = None,
    ) -> None:
        episode = {
            "input": input_data,
            "target": target,
            "output": output,
            "loss": float(loss),
            "attention": attention_map or {},
        }
        self.buffer.append(episode)

    def sample(self, n: int = 16) -> List[Dict[str, Any]]:
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[int(i)] for i in indices]

    def best_experiences(self, top_k: int = 8) -> List[Dict[str, Any]]:
        if len(self.buffer) == 0:
            return []
        sorted_eps = sorted(self.buffer, key=lambda e: e["loss"])
        return sorted_eps[: min(top_k, len(sorted_eps))]

    def __len__(self) -> int:
        return len(self.buffer)
