"""
CognitiveGraph - Mente Emergente Modular
Red de bloques cognitivos conectados que pueden colaborar y comunicarse.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from src.autograd.value import Value
from src.core.cognitive_block import CognitiveBlock


class CognitiveGraph:
    """
    Red modular de CognitiveBlocks interconectados.
    Cada bloque puede recibir señales de otros, generar salidas,
    y compartir contexto (memoria colectiva).
    """

    def __init__(self):
        self.blocks: Dict[str, CognitiveBlock] = {}
        self.connections: Dict[str, List[str]] = {}  # {block_name: [input_block_names]}
        self.shared_memory: Dict[str, List[Value]] = {}

    def add_block(self, name: str, block: CognitiveBlock) -> None:
        """Agrega un bloque cognitivo al grafo."""
        self.blocks[name] = block
        self.connections[name] = []

    def connect(self, src: str, dest: str) -> None:
        """Conecta la salida del bloque src a la entrada del bloque dest."""
        assert src in self.blocks and dest in self.blocks, f"Bloques {src} o {dest} no existen"
        self.connections[dest].append(src)

    def forward(self, inputs: Dict[str, List[float]]) -> Dict[str, List[Value]]:
        """
        Ejecuta un paso de razonamiento en toda la red.
        Cada bloque toma entradas desde sus conexiones y/o entradas externas.
        """
        outputs: Dict[str, List[Value]] = {}

        # 1️⃣ Cargar entradas externas
        for name, vals in inputs.items():
            outputs[name] = [Value(v) for v in vals]

        # 2️⃣ Propagación entre bloques conectados
        for name, block in self.blocks.items():
            # Combinar entradas de las conexiones y/o externas
            in_signals: List[Value] = []
            for src in self.connections.get(name, []):
                if src in outputs:
                    in_signals += outputs[src]

            # Si tiene entrada externa definida
            if name in outputs and not in_signals:
                in_signals = outputs[name]

            if in_signals:
                out_vals = block.forward(in_signals)
                outputs[name] = out_vals

        # 3️⃣ Guardar memoria compartida (contexto global)
        for name, block in self.blocks.items():
            self.shared_memory[name] = block.perceiver.memory.output()

        return outputs

    def step(self, inputs: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Ejecuta un paso completo y devuelve las salidas numéricas."""
        outputs = self.forward(inputs)
        return {k: [v.data for v in vals] for k, vals in outputs.items()}

    def summary(self):
        """Muestra un resumen de la estructura del grafo."""
        print("=== CognitiveGraph Summary ===")
        for name, conns in self.connections.items():
            print(f"Block '{name}' <- inputs from {conns}")
        print("==============================")

    def reset_all(self) -> None:
        """Resetea la memoria de todos los bloques."""
        for block in self.blocks.values():
            block.perceiver.reset()
            if hasattr(block, 'reasoner') and hasattr(block.reasoner, 'reset'):
                block.reasoner.reset()

    def get_memory_state(self) -> Dict[str, List[float]]:
        """Obtiene el estado actual de la memoria de todos los bloques."""
        return {name: [v.data for v in mem] for name, mem in self.shared_memory.items()}
