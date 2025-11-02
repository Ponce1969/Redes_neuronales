"""
Microneurona con soporte básico para backpropagation:
- guarda last_input, last_z (pre-activación), last_output
- permite aplicar gradientes a pesos y bias
"""

from __future__ import annotations
from typing import List
from core.activations import Activation, ACTIVATIONS
from core.utils import init_weights
from core.optimizers import Optimizer, SGD

class Neuron:
    def __init__(self, n_inputs: int, activation: str = "sigmoid", optimizer: Optimizer | None = None):
        self.n_inputs = n_inputs
        self.weights: List[float] = init_weights(n_inputs)
        self.bias: float = 0.0
        self.activation: Activation = ACTIVATIONS[activation]
        self.optimizer = optimizer or SGD(lr=0.01)
        self.last_input: List[float] | None = None
        self.last_z: float | None = None
        self.last_output: float | None = None

    def forward(self, inputs: List[float]) -> float:
        assert len(inputs) == self.n_inputs, "Número de entradas incorrecto."
        self.last_input = inputs
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_z = z
        self.last_output = self.activation(z)
        return self.last_output

    def apply_gradients(self, dweights: List[float], dbias: float, lr: float | None = None) -> None:
        """
        Actualiza los pesos y bias usando el optimizador configurado.
        dweights: gradientes dL/dw para cada peso
        dbias: gradiente dL/db
        lr: learning rate (opcional, usa el del optimizador si no se proporciona)
        """
        assert len(dweights) == self.n_inputs
        
        # Usar el learning rate del optimizador si no se proporciona
        learning_rate = lr or getattr(self.optimizer, "lr", 0.01)
        
        # Actualizar pesos con el optimizador
        self.weights = self.optimizer.update(self.weights, dweights)
        
        # Actualizar bias (usando descenso simple para mantener compatibilidad)
        self.bias -= learning_rate * dbias

    def reset(self) -> None:
        self.last_input = None
        self.last_z = None
        self.last_output = None

    def __repr__(self) -> str:
        return f"<Neuron inputs={self.n_inputs}, act={self.activation.name}>"

