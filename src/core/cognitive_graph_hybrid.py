from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # compatibilidad con ejecuciones desde paquete raíz
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
    from src.core.trm_act_block import TRM_ACT_Block  # type: ignore
    from src.core.cognitive_block import CognitiveBlock  # type: ignore
    from src.core.projection_layer import ProjectionLayer  # type: ignore
    from src.core.attention.attention_router import AttentionRouter  # type: ignore
    from src.core.monitor.cognitive_monitor import CognitiveMonitor  # type: ignore
    from src.core.memory.memory_replay import MemoryReplaySystem  # type: ignore
except ModuleNotFoundError:  # ejecución con PYTHONPATH=src
    from core.autograd_numpy.tensor import Tensor  # type: ignore
    from core.trm_act_block import TRM_ACT_Block  # type: ignore
    from core.cognitive_block import CognitiveBlock  # type: ignore
    from core.projection_layer import ProjectionLayer  # type: ignore
    from core.attention.attention_router import AttentionRouter  # type: ignore
    from core.monitor.cognitive_monitor import CognitiveMonitor  # type: ignore
    from core.memory.memory_replay import MemoryReplaySystem  # type: ignore

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
        self.last_attention: Dict[str, Dict[str, np.ndarray]] = {}
        self.attn_router = AttentionRouter()
        self.monitor = CognitiveMonitor()
        self.memory_system = MemoryReplaySystem(self, self.monitor)

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

        self.attn_router.register(src, dest, src_dim, dest_dim)


    # ------------------------------------------------------------------
    # Forward mixto
    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, List[float]]) -> Dict[str, Tensor]:
        """Ejecuta un paso de razonamiento híbrido."""
        outputs: Dict[str, Tensor] = {}
        self.last_inputs = {}
        self.last_attention = {}

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
                x = np.mean(np.stack(collected), axis=0)

            attn_sources = {}
            for src in self.connections.get(name, []):
                if src in outputs:
                    tensor_data = outputs[src].data
                    if (src, name) in self.projections:
                        tensor_data = self.projections[(src, name)].forward(outputs[src]).data
                    attn_sources[src] = tensor_data

            x_attn, attn_weights = self.attn_router.route(name, x, attn_sources)
            if attn_weights:
                self.last_attention[name] = attn_weights
                x = (x + x_attn) / 2.0
                for src, weights in attn_weights.items():
                    self.monitor.track_attention(src, name, np.array(weights))

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
            self.monitor.track_activations(name, tensor_out.data)
            block.last_activation = float(np.mean(tensor_out.data))  # type: ignore[attr-defined]
            block.last_activation_vector = tensor_out.data.copy()  # type: ignore[attr-defined]

        return outputs

    def forward_with_reasoner(
        self,
        inputs: Dict[str, List[float]],
        reasoner: Any,
        mode: str = "softmax",
        top_k: int = 2,
        temp: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Ejecuta el grafo aplicando gates que decide el `reasoner`.
        
        El reasoner analiza los planes latentes (z_plan) de cada bloque y decide
        qué bloques activar (o con qué intensidad), permitiendo rutas cognitivas selectivas.
        
        Args:
            inputs: Entradas externas por nombre de bloque
            reasoner: Instancia de Reasoner que calcula los gates
            mode: 'softmax', 'topk' o 'threshold' (pasado a reasoner.decide)
            top_k: Número de bloques a activar en modo 'topk'
            temp: Temperatura para softmax
            
        Returns:
            Diccionario con las salidas de cada bloque
        """
        outputs: Dict[str, Tensor] = {}

        # 1) Obtener z_plan por bloque (si existe)
        z_list = []
        block_names = list(self.blocks.keys())
        for name in block_names:
            blk = self.blocks[name]
            plan = None

            # Prioridad 1: LatentPlannerBlock con get_last_plan()
            if hasattr(blk, "get_last_plan") and callable(blk.get_last_plan):
                try:
                    plan = blk.get_last_plan()
                except Exception:
                    plan = None

            # Prioridad 2: TRM_ACT_Block con z
            if plan is None and hasattr(blk, "z"):
                plan = blk.z.data if hasattr(blk.z, "data") else blk.z

            # Prioridad 3: last_activation guardada en forward anterior
            if plan is None and hasattr(blk, "last_activation_vector"):
                plan = blk.last_activation_vector

            # Fallback: ceros
            if plan is None:
                plan = np.zeros((1, 1), dtype=np.float32)

            z_list.append(np.asarray(plan).reshape(1, -1))

        # 2) Reasoner decide weights por índice de bloque
        gates = reasoner.decide(z_list, mode=mode, top_k=top_k, temp=temp)

        # 3) Ejecutar bloques en orden, escalando su entrada por gate
        self.last_inputs = {}
        self.last_attention = {}

        for idx, (name, block) in enumerate(self.blocks.items()):
            gate = gates.get(idx, 1.0)

            # Recolectar inputs como en forward normal
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
                x = np.mean(np.stack(collected), axis=0)

            # Aplicar atención si hay fuentes disponibles
            attn_sources = {}
            for src in self.connections.get(name, []):
                if src in outputs:
                    tensor_data = outputs[src].data
                    if (src, name) in self.projections:
                        tensor_data = self.projections[(src, name)].forward(outputs[src]).data
                    attn_sources[src] = tensor_data

            x_attn, attn_weights = self.attn_router.route(name, x, attn_sources)
            if attn_weights:
                self.last_attention[name] = attn_weights
                x = (x + x_attn) / 2.0
                for src, weights in attn_weights.items():
                    self.monitor.track_attention(src, name, np.array(weights))

            # ✨ Aplicar gate: escalar entrada del bloque (gating selectivo)
            x = x * gate

            input_tensor = Tensor(x)
            self.last_inputs[name] = input_tensor

            # Forward del bloque según tipo
            if isinstance(block, TRM_ACT_Block):
                tensor_out = block.forward(input_tensor)
            elif isinstance(block, CognitiveBlock) or block.__class__.__name__ == "CognitiveBlock":
                value_inputs = [Value(float(v)) for v in input_tensor.data.flatten()]
                value_outputs = block.forward(value_inputs)
                arr = np.array([v.data for v in value_outputs], dtype=np.float32).reshape(
                    1, -1
                )
                tensor_out = Tensor(arr)
            else:
                raise TypeError(f"Tipo de bloque no soportado: {type(block)}")

            outputs[name] = tensor_out
            self.monitor.track_activations(name, tensor_out.data)
            block.last_activation = float(np.mean(tensor_out.data))  # type: ignore[attr-defined]
            block.last_activation_vector = tensor_out.data.copy()  # type: ignore[attr-defined]

            # Guardar gate aplicado para visualización
            if hasattr(self, "last_gates"):
                self.last_gates[name] = gate  # type: ignore[attr-defined]
            else:
                self.last_gates = {name: gate}  # type: ignore[attr-defined]

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
            if hasattr(block, "last_activation"):
                block.last_activation = 0.0  # type: ignore[attr-defined]
            if hasattr(block, "last_activation_vector"):
                block.last_activation_vector = None  # type: ignore[attr-defined]

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
