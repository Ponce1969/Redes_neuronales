"""
Celda de memoria diferenciable para Fase 7.
Mantiene estado interno con decaimiento y actualización suave.
"""

from __future__ import annotations
from typing import List
from autograd.value import Value


class MemoryCell:
    """
    Celda de memoria diferenciable con decaimiento temporal.
    Similar a una GRU simplificada o LSTM cell.
    """
    
    def __init__(self, size: int = 1, decay: float = 0.9):
        self.size = size
        self.decay = decay
        self.state = [Value(0.0) for _ in range(size)]
        
    def reset(self) -> None:
        """Resetear memoria a ceros"""
        self.state = [Value(0.0) for _ in range(self.size)]
    
    def update(self, new_input: List[Value]) -> List[Value]:
        """
        Actualización suave: h_t = decay * h_{t-1} + (1-decay) * input
        """
        assert len(new_input) == len(self.state), \
            f"Dimensión incorrecta: {len(new_input)} != {len(self.state)}"
        
        new_state = []
        for h_prev, x in zip(self.state, new_input):
            h_new = h_prev * self.decay + x * (1.0 - self.decay)
            new_state.append(h_new)
        
        self.state = new_state
        return self.state
    
    def output(self) -> List[Value]:
        """Retorna el estado actual"""
        return self.state
    
    def __repr__(self):
        return f"MemoryCell(size={self.size}, decay={self.decay})"
