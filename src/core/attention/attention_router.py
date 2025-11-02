from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from src.core.attention.attention_layer import CognitiveAttentionLayer  # type: ignore
except ModuleNotFoundError:
    from core.attention.attention_layer import CognitiveAttentionLayer  # type: ignore


class AttentionRouter:
    """Coordina capas de atención entre pares de bloques del grafo cognitivo."""

    def __init__(self) -> None:
        self.attn_layers: Dict[Tuple[str, str], CognitiveAttentionLayer] = {}

    def register(self, src: str, dest: str, dim_src: int, dim_dest: int) -> None:
        if (src, dest) not in self.attn_layers:
            self.attn_layers[(src, dest)] = CognitiveAttentionLayer(dim_dest, dim_dest, dim_dest)

    def route(
        self,
        dest_name: str,
        dest_state: np.ndarray,
        inputs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        weighted_inputs = []
        weights_summary: Dict[str, np.ndarray] = {}

        for (src, dest), layer in self.attn_layers.items():
            if dest != dest_name or src not in inputs:
                continue
            context, weights = layer.forward(dest_state, [inputs[src]], [inputs[src]])
            weighted_inputs.append(context)
            weights_summary[src] = weights

        if not weighted_inputs:
            return np.zeros_like(dest_state), {}

        combined = np.mean(np.stack(weighted_inputs), axis=0)
        return combined, weights_summary
