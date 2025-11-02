"""
Macro-neurona cognitiva para Fase 7.
Combina micro-neuronas con memoria y gating para razonamiento secuencial.
"""

from __future__ import annotations
from typing import List
import random
from autograd.value import Value
from autograd.functional import linear
# Usar tanh directo desde Value
from core.memory_cell import MemoryCell


class MacroNeuron:
    """
    Macro-neurona que integra:
    - Micro-neuronas internas
    - Memoria persistente
    - Gating para control de información
    """
    
    def __init__(self, n_inputs: int, n_hidden: int = 3, decay: float = 0.9):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.decay = decay
        
        # Pesos para entrada
        self.input_weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(n_inputs)]
        
        # Pesos para memoria
        self.memory_weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(n_hidden)]
        
        # Pesos ocultos para gate
        self.gate_weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(n_inputs)]
        self.gate_bias = Value(0.0)
        
        # Bias principal
        self.bias = Value(0.0)
        
        # Celda de memoria
        self.memory = MemoryCell(size=n_hidden, decay=decay)
    
    def forward(self, inputs: List[Value]) -> List[Value]:
        """
        Forward pass que combina input actual + memoria previa
        """
        h_prev = self.memory.output()
        
        # 1. Cálculo de gate (cuánto recordar vs actualizar)
        gate_sum = sum(w * x for w, x in zip(self.gate_weights, inputs)) + self.gate_bias
        gate = gate_sum.tanh()
        
        # 2. Candidato a nueva memoria
        combined = inputs + h_prev
        weights = self.input_weights + self.memory_weights
        
        # Calcular candidato para cada dimensión
        h_new = []
        one = Value(1.0)
        
        # Para cada dimensión de la memoria
        for h in h_prev:
            # Calcular input para esta dimensión
            input_sum = sum(w * x for w, x in zip(self.input_weights, inputs))
            memory_sum = sum(w * h_prev[j] for j, w in enumerate(self.memory_weights))
            candidate = (input_sum + memory_sum + self.bias).tanh()
            
            # Mezcla con gate
            h_new.append(h * gate + candidate * (one - gate))
        
        # 3. Actualizar memoria y retornar
        self.memory.update(h_new)
        return self.memory.output()
    
    def reset(self) -> None:
        """Resetear memoria para nueva secuencia"""
        self.memory.reset()
    
    def __repr__(self):
        return f"MacroNeuron(inputs={self.n_inputs}, hidden={self.n_hidden})"
