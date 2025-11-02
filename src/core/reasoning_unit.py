"""
Pequeño módulo que toma percepciones + memoria y produce inferencias simples.
"""

from __future__ import annotations
from typing import List
from src.autograd.value import Value


class ReasoningUnit:
    """
    Combina percepción (inputs) y memoria (contexto) para generar una inferencia intermedia.
    """
    def __init__(self, n_inputs: int, n_memory: int, n_out: int):
        self.weights_in = [Value(0.1) for _ in range(n_inputs)]
        self.weights_mem = [Value(0.1) for _ in range(n_memory)]
        self.bias = Value(0.0)
        self.n_out = n_out

    def forward(self, inputs: List[Value], memory: List[Value]) -> List[Value]:
        combined = inputs + memory
        out = []
        for _ in range(self.n_out):
            val = Value(0.0)
            for w, x in zip(self.weights_in + self.weights_mem, combined):
                val = val + w * x
            val = val + self.bias
            val = val.tanh()
            out.append(val)
        return out
