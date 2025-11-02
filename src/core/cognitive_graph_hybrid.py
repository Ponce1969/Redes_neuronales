from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # compatibilidad con ejecuciones desde paquete raíz
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
    from src.core.trm_act_block import TRM_ACT_Block  # type: ignore
    from src.core.cognitive_block import CognitiveBlock  # type: ignore
    from src.core.projection_layer import ProjectionLayer  # type: ignore
except ModuleNotFoundError:  # ejecución con PYTHONPATH=src
    from core.autograd_numpy.tensor import Tensor  # type: ignore
    from core.trm_act_block import TRM_ACT_Block  # type: ignore
    from core.cognitive_block import CognitiveBlock  # type: ignore
    from core.projection_layer import ProjectionLayer  # type: ignore

try:
    from src.autograd.value import Value  # type: ignore
except ModuleNotFoundError:
    from autograd.value import Value  # type: ignore


class CognitiveGraphHybrid:
    """Grafo cognitivo híbrido que orquesta bloques clásicos y TRM adaptativos."""

    def __init__(self) -> None:
        self.blocks: Dict[str, Any] = {}
        self.connections: Dict[str, List[str]] = {}
        self.projections: Dict[tuple[str, str], ProjectionLayer] = {}
        self.last_inputs: Dict[str, Tensor] = {}

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

        src_dim = self._get_output_dim(self.blocks[src])
        dest_dim = self._get_input_dim(self.blocks[dest])

        if src_dim != dest_dim:
            self.projections[(src, dest)] = ProjectionLayer(src_dim, dest_dim)


    # ------------------------------------------------------------------
    # Forward mixto
    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, List[float]]) -> Dict[str, Tensor]:
        """Ejecuta un paso de razonamiento híbrido."""
        outputs: Dict[str, Tensor] = {}
        self.last_inputs = {}

        for name, block in self.blocks.items():
            collected: List[np.ndarray] = []

            # Señales internas
            for src in self.connections.get(name, []):
                if src in outputs:
                    data = outputs[src]
                    out_tensor = data
                    if (src, name) in self.projections:
                        proj = self.projections[(src, name)]
                        out_tensor = proj.forward(out_tensor)
                    collected.append(out_tensor.data)

            # Entradas externas
            if name in inputs:
                collected.append(np.array(inputs[name], dtype=np.float32).reshape(1, -1))

            if not collected:
                in_dim = self._infer_input_dim(block)
                x = np.zeros((1, in_dim), dtype=np.float32)
            else:
                x = np.concatenate(collected, axis=1)

            input_tensor = Tensor(x)
            self.last_inputs[name] = input_tensor

            if isinstance(block, TRM_ACT_Block):
                tensor_out = block.forward(input_tensor)
            elif isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
                value_inputs = [Value(float(v)) for v in input_tensor.data.flatten()]
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
        if isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
            return int(getattr(block, "input_size", getattr(block.perceiver, "n_inputs", 1)))
        return 1

    @staticmethod
    def _get_output_dim(block: Any) -> int:
        if isinstance(block, TRM_ACT_Block):
            return int(block.W_out.data.shape[1])
        if isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
            return int(getattr(block, "output_size", 1))
        return 1

    @staticmethod
    def _get_input_dim(block: Any) -> int:
        if isinstance(block, TRM_ACT_Block):
            return int(block.W_in.data.shape[0])
        if isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
            return int(getattr(block, "input_size", 1))
        return 1
