"""
Reasoner: Controlador que decide gates (pesos de activación) para cada bloque del grafo.
Implementación ligera en NumPy compatible con estrategias evolutivas.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class Reasoner:
    """
    Controller que decide gates (pesos de activación) para cada bloque del grafo.
    
    Implementación ligera en NumPy:
      - Small MLP: z_concat -> hidden -> logits (n_blocks) -> softmax or top-k gating
      - Puede ser optimizado vía estrategias evolutivas (mutación simple).
    
    Args:
        n_inputs: Tamaño del vector concatenado de z_plan por bloque
        n_hidden: Número de neuronas en la capa oculta
        n_blocks: Número de bloques en el grafo cognitivo
        seed: Semilla para reproducibilidad
    """

    def __init__(
        self, n_inputs: int, n_hidden: int, n_blocks: int, seed: Optional[int] = None
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks

        # MLP params
        self.W1 = rng.normal(scale=0.1, size=(n_inputs, n_hidden)).astype(np.float32)
        self.b1 = np.zeros((1, n_hidden), dtype=np.float32)
        self.W2 = rng.normal(scale=0.1, size=(n_hidden, n_blocks)).astype(np.float32)
        self.b2 = np.zeros((1, n_blocks), dtype=np.float32)

    def _mlp(self, x: np.ndarray) -> np.ndarray:
        """Forward pass del MLP: x -> hidden -> logits."""
        h = np.tanh(x @ self.W1 + self.b1)  # (1, n_hidden)
        logits = h @ self.W2 + self.b2  # (1, n_blocks)
        return logits.squeeze()

    def decide(
        self,
        z_per_block: List[np.ndarray],
        mode: str = "softmax",
        top_k: int = 2,
        temp: float = 1.0,
    ) -> Dict[int, float]:
        """
        Devuelve un diccionario {block_index: weight}, manteniendo el orden del input.
        
        Args:
            z_per_block: Lista de arrays (1, z_dim) o scalars; serán concatenados
            mode: 'softmax' (distribución sobre todos), 'topk' (sparse: 1 for top_k else eps), 
                  'threshold' (>=epsilon)
            top_k: Número de bloques a activar en modo 'topk'
            temp: Temperatura para softmax
            
        Returns:
            Diccionario con índice de bloque y su peso de activación
        """
        # Concatenar: convertir cada z a vector plano y concatenar
        flat = []
        for z in z_per_block:
            arr = np.asarray(z).reshape(-1)
            flat.append(arr)

        if len(flat) == 0:
            x = np.zeros((1, self.n_inputs), dtype=np.float32)
        else:
            x = np.concatenate(flat).astype(np.float32)
            # Si el tamaño concatenado difiere de n_inputs, pad o truncate
            if x.shape[0] < self.n_inputs:
                pad = np.zeros(self.n_inputs - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad])
            elif x.shape[0] > self.n_inputs:
                x = x[: self.n_inputs]
        x = x.reshape(1, -1)

        logits = self._mlp(x) / max(1e-6, temp)

        if mode == "softmax":
            ex = np.exp(logits - np.max(logits))
            probs = ex / (np.sum(ex) + 1e-9)
            weights = probs.tolist()
        elif mode == "topk":
            weights = [0.0] * self.n_blocks
            idxs = np.argsort(-logits)[:top_k]
            for i in idxs:
                weights[int(i)] = 1.0 / top_k
        elif mode == "threshold":
            thr = 0.1
            weights = (logits > thr).astype(float).tolist()
            s = sum(weights)
            if s > 0:
                weights = [w / s for w in weights]
            else:
                # Fallback a softmax si ninguno supera threshold
                ex = np.exp(logits - np.max(logits))
                probs = ex / (np.sum(ex) + 1e-9)
                weights = probs.tolist()
        else:
            raise ValueError("mode must be 'softmax', 'topk' or 'threshold'")

        # Retornar mapping por índice
        return {i: float(weights[i]) for i in range(self.n_blocks)}

    def mutate(self, scale: float = 0.02, seed: Optional[int] = None) -> "Reasoner":
        """
        Devuelve una copia mutada del Reasoner (para evolución).
        
        Args:
            scale: Magnitud de la mutación gaussiana
            seed: Semilla para reproducibilidad
            
        Returns:
            Nuevo Reasoner con parámetros mutados
        """
        rng = np.random.default_rng(seed)
        child = Reasoner(self.n_inputs, self.n_hidden, self.n_blocks)
        # Copiar params y mutar
        child.W1 = (self.W1 + rng.normal(scale=scale, size=self.W1.shape)).astype(
            np.float32
        )
        child.b1 = (self.b1 + rng.normal(scale=scale, size=self.b1.shape)).astype(
            np.float32
        )
        child.W2 = (self.W2 + rng.normal(scale=scale, size=self.W2.shape)).astype(
            np.float32
        )
        child.b2 = (self.b2 + rng.normal(scale=scale, size=self.b2.shape)).astype(
            np.float32
        )
        return child

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Devuelve el estado serializable del Reasoner."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        """Carga el estado desde un diccionario."""
        self.W1 = sd["W1"].astype(np.float32)
        self.b1 = sd["b1"].astype(np.float32)
        self.W2 = sd["W2"].astype(np.float32)
        self.b2 = sd["b2"].astype(np.float32)
