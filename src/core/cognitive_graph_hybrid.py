from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # compatibilidad con ejecuciones desde paquete raíz
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
    from src.core.trm_act_block import TRM_ACT_Block  # type: ignore
    from src.core.cognitive_block import CognitiveBlock  # type: ignore
except ModuleNotFoundError:  # ejecución con PYTHONPATH=src
    from core.autograd_numpy.tensor import Tensor  # type: ignore
    from core.trm_act_block import TRM_ACT_Block  # type: ignore
    from core.cognitive_block import CognitiveBlock  # type: ignore

try:
    from src.autograd.value import Value  # type: ignore
except ModuleNotFoundError:
    from autograd.value import Value  # type: ignore


class CognitiveGraphHybrid:
    """Grafo cognitivo híbrido que orquesta bloques clásicos y TRM adaptativos."""

    def __init__(self) -> None:
        self.blocks: Dict[str, Any] = {}
        self.connections: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Gestión de nodos y conexiones
    # ------------------------------------------------------------------
    def add_block(self, name: str, block: Any) -> None:
        if name in self.blocks:
            raise ValueError(f"Block '{name}' ya existe en el grafo.")
        self.blocks[name] = block
        self.connections[name] = []

    def connect(self, src: str, dest: str) -> None:
        if src not in self.blocks or dest not in self.blocks:
            raise KeyError(f"Bloques desconocidos: {src}, {dest}")
        self.connections[dest].append(src)

    # ------------------------------------------------------------------
    # Forward mixto
    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, List[float]]) -> Dict[str, Tensor]:
        """Ejecuta un paso de razonamiento híbrido."""
        outputs: Dict[str, Tensor] = {}

        for name, block in self.blocks.items():
            collected: List[np.ndarray] = []

            # Señales internas
            for src in self.connections.get(name, []):
                if src in outputs:
                    data = outputs[src]
                    if isinstance(block, CognitiveBlock):
                        arr = data.data
                        collected.append(arr)
                    else:
                        collected.append(data.data)

            # Entradas externas
            if name in inputs:
                collected.append(np.array(inputs[name], dtype=np.float32).reshape(1, -1))

            if not collected:
                in_dim = self._infer_input_dim(block)
                x = np.zeros((1, in_dim), dtype=np.float32)
            else:
                x = np.concatenate(collected, axis=1)

            if isinstance(block, TRM_ACT_Block):
                tensor_out = block.forward(Tensor(x))
            elif isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
                value_inputs = [Value(float(v)) for v in x.flatten()]
                value_outputs = block.forward(value_inputs)
                arr = np.array([v.data for v in value_outputs], dtype=np.float32).reshape(1, -1)
                tensor_out = Tensor(arr)
            else:
                raise TypeError(f"Tipo de bloque no soportado: {type(block)}")

            outputs[name] = tensor_out

        return outputs

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def reset_states(self) -> None:
        for block in self.blocks.values():
            if isinstance(block, TRM_ACT_Block):
                block.z = Tensor(np.zeros_like(block.z.data))
            elif hasattr(block, "perceiver") and hasattr(block.perceiver, "reset"):
                block.perceiver.reset()

    def summary(self) -> None:
        print("=== CognitiveGraphHybrid Summary ===")
        for name, block in self.blocks.items():
            block_type = type(block).__name__
            if isinstance(block, TRM_ACT_Block):
                info = f"(recursivo, hidden={block.W_in.data.shape[1]}, steps={block.max_steps})"
            elif isinstance(block, CognitiveBlock):
                hidden = getattr(block.perceiver, "n_hidden", getattr(block.perceiver, "n_hidden", "?"))
                info = f"(clásico, hidden={hidden})"
            else:
                info = "(tipo desconocido)"
            print(f"Block '{name}' -> {block_type} {info}")
            print(f"   Conectado desde: {self.connections.get(name, [])}")
        print("===================================")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_input_dim(block: Any) -> int:
        if isinstance(block, TRM_ACT_Block):
            return int(block.W_in.data.shape[0])
        if isinstance(block, CognitiveBlock):
            return int(getattr(block.perceiver, "n_inputs", 1))
        return 1
