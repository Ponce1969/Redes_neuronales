"""
Definición de una capa de neuronas.
Cada capa contiene múltiples microneuronas y gestiona la propagación.
"""

from __future__ import annotations
from typing import List
from core.neuron import Neuron


class Layer:
    """
    Representa una capa de neuronas densamente conectadas.

    Atributos:
        n_inputs: número de entradas que recibe cada neurona.
        n_neurons: número total de neuronas en la capa.
        neurons: lista de objetos Neuron.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "sigmoid"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.neurons: List[Neuron] = [
            Neuron(n_inputs, activation=activation) for _ in range(n_neurons)
        ]

    def forward(self, inputs: List[float]) -> List[float]:
        """Propaga el vector de entrada a través de todas las neuronas de la capa."""
        return [neuron.forward(inputs) for neuron in self.neurons]

    def reset(self) -> None:
        """Resetea el estado de todas las neuronas."""
        for neuron in self.neurons:
            neuron.reset()

    def __repr__(self) -> str:
        return f"<Layer neurons={self.n_neurons}, inputs={self.n_inputs}>"


