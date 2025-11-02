"""
Macro-neurona cognitiva para Fase 7 y 8.
Combina micro-neuronas con memoria y gating para razonamiento secuencial.
"""

from __future__ import annotations
from typing import List
import random
from src.autograd.value import Value
from src.core.memory_cell import MemoryCell


class MacroNeuron:
    """
    Macro-neurona cognitiva corregida para Fase 7-9.
    Combina micro-neuronas con memoria y gating para razonamiento secuencial.
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
        gate_sum = Value(0.0)
        for w, x in zip(self.gate_weights, inputs):
            gate_sum = gate_sum + w * x
        gate_sum = gate_sum + self.gate_bias
        gate = gate_sum.tanh()
        
        # 2. Candidato a nueva memoria para cada dimensión
        h_new = []
        one = Value(1.0)
        
        for i in range(len(h_prev)):
            # Input para esta dimensión
            input_sum = Value(0.0)
            for w, x in zip(self.input_weights, inputs):
                input_sum = input_sum + w * x
            
            # Memoria para esta dimensión - CORREGIDO
            memory_sum = Value(0.0)
            for j, w in enumerate(self.memory_weights):
                memory_sum = memory_sum + w * h_prev[j]
            
            candidate = (input_sum + memory_sum + self.bias).tanh()
            
            # Mezcla con gate
            h_new_val = h_prev[i] * gate + candidate * (one - gate)
            h_new.append(h_new_val)
        
        # 3. Actualizar memoria y retornar
        self.memory.update(h_new)
        return self.memory.output()
    
    def reset(self) -> None:
        """Resetear memoria para nueva secuencia"""
        self.memory.reset()
    
    def __repr__(self):
        return f"MacroNeuron(inputs={self.n_inputs}, hidden={self.n_hidden})"
