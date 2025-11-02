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
            if not (0 <= self.decoder_layer_idx < len(layer_sizes) - 1):
                raise ValueError(f"decoder_layer_idx debe estar entre 0 y {len(layer_sizes) - 2}")
            
            # Ajustar dimensiones: incrementar entrada de la capa objetivo
            ls = list(layer_sizes)
            ls[self.decoder_layer_idx] = ls[self.decoder_layer_idx] + self.z_proj_dim
            self.layer_sizes = ls
            
            # Crear projector
            self.projector = z_config.get("projector") or latent_mod.LatentProjector(self.z_dim, self.z_proj_dim)
        else:
            self.z_dim = 0
            self.decoder_layer_idx = -1
            self.z_proj_dim = 0
            self.layer_sizes = list(layer_sizes)
            self.projector = None

        # inicializamos capas con layer_sizes ajustados
        self.layers: List[Layer] = [
            Layer(self.layer_sizes[i], self.layer_sizes[i + 1], activation=activation)
            for i in range(len(self.layer_sizes) - 1)
        ]

    def forward(self, inputs: List[float], z: Optional[List[float]] = None) -> List[float]:
        """
        Propaga las entradas a través de la red.
        Si z se proporciona y la red fue inicializada con z_config, se proyecta y concatena
        a las entradas de la capa `decoder_layer_idx`.
        """
        x = list(inputs)  # copia por seguridad
        
        # Validar dimensiones iniciales
        if len(x) != self.layers[0].n_inputs:
            if z is not None and self.z_config is not None and self.decoder_layer_idx == 0:
                # Caso especial: z se concatena con inputs iniciales
                expected = self.layers[0].n_inputs - self.z_proj_dim
                if len(x) != expected:
                    raise ValueError(f"Inputs esperados: {expected}, recibidos: {len(x)}")
            else:
                expected = self.layers[0].n_inputs
                if len(x) != expected:
                    raise ValueError(f"Inputs esperados: {expected}, recibidos: {len(x)}")
        
        for idx, layer in enumerate(self.layers):
            if z is not None and idx == self.decoder_layer_idx:
                # Validar z
                if self.projector is None:
                    raise ValueError("Projector no inicializado")
                
                z_proj = self.projector.project(z)
                
                # Validar que la concatenación sea posible
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
        # Validar dimensiones de entrada
        expected_inputs = self.layers[0].n_inputs
        if z is not None and self.decoder_layer_idx == 0:
            expected_inputs -= self.z_proj_dim
        
        if len(inputs) != expected_inputs:
            raise ValueError(
                f"Dimensiones incorrectas: esperado {expected_inputs}, recibido {len(inputs)}"
            )
        
        # Forward con validación de z
        if z is not None and self.decoder_layer_idx == 0:
            if self.projector is None:
                raise ValueError("Projector no inicializado para z")
            z_proj = self.projector.project(z)
            full_inputs = inputs + z_proj
        else:
            full_inputs = inputs
        
        # Validar dimensiones para la primera capa
        if len(full_inputs) != self.layers[0].n_inputs:
            raise ValueError(
                f"Dimensión después de concatenar z: {len(full_inputs)}, esperado: {self.layers[0].n_inputs}"
            )
        
        # Forward propagation
        outputs = self.forward(full_inputs, z=z if self.decoder_layer_idx > 0 else None)
        
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
        # Forward
        outputs = self.forward(inputs)

        # Gradiente inicial: dL/dy para cada output
        dL_dy = loss_grad_fn(outputs, targets)  # lista de tamaño output_dim

        # Backpropagation: calculamos deltas y actualizamos capa por capa (hacia atrás)
        next_deltas: List[float] | None = None
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            layer_deltas: List[float] = [0.0] * layer.n_neurons

            if layer_idx == len(self.layers) - 1:
                # capa de salida: dL/dz = (dL/dy) * activation'(z)
                for j, neuron in enumerate(layer.neurons):
                    assert neuron.last_z is not None and neuron.last_output is not None
                    act_deriv = neuron.activation.derivative(neuron.last_z)
                    layer_deltas[j] = dL_dy[j] * act_deriv
            else:
                # capa oculta: dL/dz_i = activation'(z_i) * sum_j w_ij_next * delta_next_j
                next_layer = self.layers[layer_idx + 1]
                for i, neuron in enumerate(layer.neurons):
                    assert neuron.last_z is not None
                    sum_w_delta = 0.0
                    for j, next_neuron in enumerate(next_layer.neurons):
                        sum_w_delta += next_neuron.weights[i] * next_deltas[j]
                    layer_deltas[i] = neuron.activation.derivative(neuron.last_z) * sum_w_delta

            # Aplicar gradientes en cada neurona
            for neuron_idx, neuron in enumerate(layer.neurons):
                assert neuron.last_input is not None
                delta = layer_deltas[neuron_idx]
                # grad w.r.t. weights: dL/dw_i = delta * input_i
                dweights = [delta * x for x in neuron.last_input]
                dbias = delta
                neuron.apply_gradients(dweights, dbias, lr)

            next_deltas = layer_deltas

        # Return loss scalar if user wants to compute it elsewhere (trainer will call loss_fn)
        # Here we don't compute loss value (caller can compute it)
        return 0.0
