from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:  # compatibilidad con ejecuciones desde paquete raíz
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
except ModuleNotFoundError:
    from core.autograd_numpy.tensor import Tensor  # type: ignore

try:
    from src.core.attention.utils import ensure_2d, safe_softmax  # type: ignore
except ModuleNotFoundError:
    from core.attention.utils import ensure_2d, safe_softmax  # type: ignore


class CognitiveAttentionLayer:
    """Capa de atención cognitiva con proyecciones simples Query-Key-Value."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int) -> None:
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.W_q = Tensor(np.random.randn(dim_q, dim_k) * 0.1)
        self.W_k = Tensor(np.random.randn(dim_k, dim_k) * 0.1)
        self.W_v = Tensor(np.random.randn(dim_v, dim_v) * 0.1)
        self.scale = np.sqrt(dim_k)

    def forward(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = ensure_2d(query) @ self.W_q.data
        K = np.stack([ensure_2d(k) @ self.W_k.data for k in keys])
        V = np.stack([ensure_2d(v) @ self.W_v.data for v in values])

        attn_logits = (Q @ K.transpose(0, 2, 1)).squeeze(0) / self.scale
        attn_weights = safe_softmax(attn_logits.squeeze(-1))

        context = np.sum(attn_weights[:, None] * V.squeeze(1), axis=0)
        return context.reshape(1, -1), attn_weights

    def __repr__(self) -> str:
        return (
            "CognitiveAttentionLayer(" f"q={self.W_q.data.shape}, "
            f"k={self.W_k.data.shape}, v={self.W_v.data.shape})"
        )
