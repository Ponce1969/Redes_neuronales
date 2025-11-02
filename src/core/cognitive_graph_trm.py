from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from autograd_numpy.tensor import Tensor
from core.trm_act_block import TRM_ACT_Block


class CognitiveGraphTRM:
    """
    CognitiveGraph especializado para TRM_ACT blocks.
    Cada nodo es un TRM_ACT_Block (razonador recursivo adaptativo).
    Las conexiones son direcciones simples: salida(src) -> entrada(dest).
    """

    def __init__(self) -> None:
        self.blocks: Dict[str, TRM_ACT_Block] = {}
        self.connections: Dict[str, List[str]] = {}
        self.projections: Dict[str, Tuple[int, int]] = {}

    def add_block(self, name: str, block: TRM_ACT_Block) -> None:
        if name in self.blocks:
            raise ValueError(f"Block '{name}' already exists in graph.")
        self.blocks[name] = block
        self.connections[name] = []

    def connect(self, src: str, dest: str) -> None:
        if src not in self.blocks:
            raise KeyError(f"Source block '{src}' not found.")
        if dest not in self.blocks:
            raise KeyError(f"Destination block '{dest}' not found.")
        self.connections[dest].append(src)

    def forward_step(self, external_inputs: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        """Ejecuta un paso de razonamiento en todo el grafo."""
        outputs: Dict[str, Tensor] = {}

        for name, arr in external_inputs.items():
            if name not in self.blocks:
                raise KeyError(f"External input provided for unknown block '{name}'.")
            outputs[name] = Tensor(np.array(arr, dtype=np.float32))

        for name, block in self.blocks.items():
            collected: List[np.ndarray] = []
            for src in self.connections.get(name, []):
                if src in outputs:
                    collected.append(outputs[src].data)
            if name in external_inputs and name not in outputs:
                collected.append(np.array(external_inputs[name], dtype=np.float32))

            if not collected:
                n_in = block.W_in.data.shape[0]
                in_arr = np.zeros((1, n_in), dtype=np.float32)
            else:
                in_arr = np.concatenate(collected, axis=1)

            out_tensor = block.forward(Tensor(in_arr))
            outputs[name] = out_tensor

        return outputs

    def step_numeric(self, external_inputs: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        np_inputs = {k: np.array(v, dtype=np.float32).reshape(1, -1) for k, v in external_inputs.items()}
        outs = self.forward_step(np_inputs)
        return {k: v.data for k, v in outs.items()}

    def reset_states(self) -> None:
        for block in self.blocks.values():
            if hasattr(block, "z"):
                block.z = Tensor(np.zeros_like(block.z.data))

    def summary(self) -> None:
        print("=== CognitiveGraphTRM Summary ===")
        for name, block in self.blocks.items():
            in_dim = block.W_in.data.shape[0]
            hidden = block.W_in.data.shape[1]
            steps = getattr(block, "max_steps", getattr(block, "n_steps", "N/A"))
            print(f"Block '{name}': in_dim={in_dim}, hidden={hidden}, max_steps={steps}")
            print(f"  Inputs from: {self.connections.get(name, [])}")
        print("=================================")
