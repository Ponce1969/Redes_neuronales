"""
El núcleo: combina percepción, memoria y razonamiento en una estructura integrada.
"""

from __future__ import annotations
from typing import List
from src.autograd.value import Value


class CognitiveBlock:
    """
    Un bloque cognitivo completo:
    - Percibe entradas
    - Usa macro-neuronas para mantener memoria interna
    - Usa un razonador para generar conclusiones intermedias
    - Usa un decisor para producir salidas
    """

    def __init__(self, n_inputs: int, n_hidden: int = 2, n_outputs: int = 1):
        from src.core.macro_neuron import MacroNeuron
        from src.core.reasoning_unit import ReasoningUnit
        
        self.perceiver = MacroNeuron(n_inputs=n_inputs, n_hidden=n_hidden)
        self.reasoner = ReasoningUnit(n_inputs=1, n_memory=n_hidden, n_out=n_hidden)
        self.decision_weights = [Value(0.1) for _ in range(n_hidden)]
        self.decision_bias = Value(0.0)
        self.n_outputs = n_outputs

    def forward(self, inputs: List[Value]) -> List[Value]:
        # 1. Percibir estímulo y actualizar memoria
        perception = self.perceiver.forward(inputs)

        # 2. Generar inferencias (razonamiento sobre percepción + memoria)
        inference = self.reasoner.forward(inputs, perception)

        # 3. Decidir (combinación ponderada del razonamiento)
        out = []
        for _ in range(self.n_outputs):
            val = Value(0.0)
            for w, x in zip(self.decision_weights, inference):
                val = val + w * x
            val = val + self.decision_bias
            val = val.tanh()
            out.append(val)
        return out
