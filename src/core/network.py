"""
Red neuronal con soporte para variable latente z y backpropagation.
Fase 5: introducción de z para planificación interna.
"""

from __future__ import annotations
from typing import List, Optional
from core.layer import Layer
from core.neuron import Neuron
from core.activations import Activation
from core import latent as latent_mod
from core import losses
from core.optimizers import Optimizer, clone_optimizer

class NeuralNetwork:
    """
    Red neuronal multicapa con soporte opcional de variable latente z.

    Parámetros relevantes:
    - layer_sizes: lista [n_in, n_h1, ..., n_out]
    - activation: nombre de la activación por defecto
    - z_config: opcional dict:
        {
            "z_dim": int,
            "decoder_layer_idx": int,  # índice de la capa donde concatenar z
            "z_proj_dim": int,         # cuanto proyectar z antes de concatenar
            "projector": LatentProjector (opcional)
        }
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "sigmoid",
        z_config: dict | None = None,
    ):
        self.z_config = z_config
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation

        if z_config is not None:
            # Validar configuración
            required_keys = ["z_dim", "decoder_layer_idx", "z_proj_dim"]
            for key in required_keys:
                if key not in z_config:
                    raise ValueError(f"z_config debe contener '{key}'")

            self.z_dim = int(z_config["z_dim"])
            self.decoder_layer_idx = int(z_config["decoder_layer_idx"])
            self.z_proj_dim = int(z_config["z_proj_dim"])

            # Validar índice
            if not (0 <= self.decoder_layer_idx < len(self.layer_sizes) - 1):
                raise ValueError(
                    f"decoder_layer_idx debe estar entre 0 y {len(self.layer_sizes) - 2}"
                )

            # Crear projector
            self.projector = z_config.get("projector") or latent_mod.LatentProjector(
                self.z_dim, self.z_proj_dim
            )
        else:
            self.z_dim = 0
            self.decoder_layer_idx = -1
            self.z_proj_dim = 0
            self.projector = None

        # inicializamos capas calculando entradas efectivas (considerando z)
        self.layers: List[Layer] = []
        for i in range(len(self.layer_sizes) - 1):
            n_inputs = self.layer_sizes[i]
            if self.decoder_layer_idx == i and self.z_proj_dim > 0:
                n_inputs += self.z_proj_dim
            layer = Layer(n_inputs, self.layer_sizes[i + 1], activation=activation)
            self.layers.append(layer)

        self._configured_optimizer: Optional[Optimizer] = None

    def configure_optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """Asigna copias del optimizador a todas las neuronas.

        Cuando ``optimizer`` es ``None`` se dejan los optimizadores por defecto de
        cada neurona (SGD independiente)."""

        self._configured_optimizer = optimizer
        if optimizer is None:
            return

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.optimizer = clone_optimizer(optimizer)

    def forward(self, inputs: List[float], z: Optional[List[float]] = None) -> List[float]:
        """Propaga las entradas a través de la red, concatenando ``z`` automáticamente."""

        has_latent = self.z_config is not None
        z_proj: Optional[List[float]] = None

        if has_latent:
            if self.projector is None:
                raise ValueError("Projector no inicializado")
            if z is None:
                z_input = [0.0] * self.z_dim
            else:
                if len(z) != self.z_dim:
                    raise ValueError(f"Dimensión de z incorrecta: esperado {self.z_dim}, recibido {len(z)}")
                z_input = list(z)
            z_proj = self.projector.project(z_input)

        x = list(inputs)
        expected_inputs = self.layer_sizes[0]
        if len(x) != expected_inputs:
            raise ValueError(f"Inputs esperados: {expected_inputs}, recibidos: {len(x)}")

        if has_latent and self.decoder_layer_idx == 0:
            assert z_proj is not None
            if len(x) + len(z_proj) != self.layers[0].n_inputs:
                raise ValueError(
                    f"Concatenación imposible: {len(x)} + {len(z_proj)} != {self.layers[0].n_inputs}"
                )
            x = x + z_proj

        for idx, layer in enumerate(self.layers):
            if has_latent and self.decoder_layer_idx == idx and self.decoder_layer_idx > 0:
                assert z_proj is not None
                if len(x) + len(z_proj) != layer.n_inputs:
                    raise ValueError(
                        f"Concatenación imposible: {len(x)} + {len(z_proj)} != {layer.n_inputs}"
                    )
                x = x + z_proj

            x = layer.forward(x)
        return x

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def summary(self) -> None:
        print("=== Network Summary ===")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.n_inputs} -> {layer.n_neurons}")
        if self.z_dim > 0:
            print(f"Latent z_dim={self.z_dim}, decoder_layer_idx={self.decoder_layer_idx}, z_proj_dim={self.z_proj_dim}")
        print("========================")

    # ---------------- Training core: single step (forward + backward update) ----------------
    def train_step(
        self,
        inputs: List[float],
        targets: List[float],
        loss_grad_fn,  # función que dado (y_pred, y_true) devuelve lista dL/dy_i
        lr: float = 0.01,
        z: Optional[List[float]] = None,
    ) -> float:
        """
        Train step con soporte para variable latente z.
        """
        # Validar dimensiones de entrada (antes de concatenar z)
        expected_inputs = self.layer_sizes[0]
        if len(inputs) != expected_inputs:
            raise ValueError(
                f"Dimensiones incorrectas: esperado {expected_inputs}, recibido {len(inputs)}"
            )
        
        # Forward propagation con integración automática de z
        outputs = self.forward(inputs, z=z)
        
        # Gradiente de la loss
        dL_dy = loss_grad_fn(outputs, targets)
        
        # Backpropagation
        next_deltas: List[float] | None = None
        
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            layer_deltas: List[float] = [0.0] * layer.n_neurons

            if layer_idx == len(self.layers) - 1:
                # capa de salida
                for j, neuron in enumerate(layer.neurons):
                    assert neuron.last_z is not None
                    act_deriv = neuron.activation.derivative(neuron.last_z)
                    layer_deltas[j] = dL_dy[j] * act_deriv
            else:
                # capa oculta
                next_layer = self.layers[layer_idx + 1]
                for i, neuron in enumerate(layer.neurons):
                    assert neuron.last_z is not None
                    sum_w_delta = 0.0
                    for j, next_neuron in enumerate(next_layer.neurons):
                        sum_w_delta += next_neuron.weights[i] * next_deltas[j]
                    layer_deltas[i] = neuron.activation.derivative(neuron.last_z) * sum_w_delta

            # Aplicar gradientes
            for neuron_idx, neuron in enumerate(layer.neurons):
                assert neuron.last_input is not None
                delta = layer_deltas[neuron_idx]
                dweights = [delta * x for x in neuron.last_input]
                dbias = delta
                neuron.apply_gradients(dweights, dbias, lr)

            next_deltas = layer_deltas

        return losses.mse_loss(outputs, targets)
