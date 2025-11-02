from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

try:
    from src.core.autograd_numpy.tensor import Tensor  # type: ignore
    from src.core.training.losses import mse_loss  # type: ignore
    from src.core.training.optimizers import Adam, SGD, Parameter  # type: ignore
except ModuleNotFoundError:
    from core.autograd_numpy.tensor import Tensor  # type: ignore
    from core.training.losses import mse_loss  # type: ignore
    from core.training.optimizers import Adam, SGD, Parameter  # type: ignore

try:
    from src.autograd.value import Value  # type: ignore
except ModuleNotFoundError:
    from autograd.value import Value  # type: ignore

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.trm_act_block import TRM_ACT_Block
from core.cognitive_block import CognitiveBlock

BlockType = CognitiveBlock | TRM_ACT_Block


def _to_tensor(value: np.ndarray | float | List[float]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(np.array(value, dtype=np.float32))


class GraphTrainer:
    """Entrena un CognitiveGraphHybrid combinando pérdidas locales y globales."""

    def __init__(
        self,
        graph: CognitiveGraphHybrid,
        lr: float = 0.01,
        loss_fn=mse_loss,
        optimizer_cls=Adam,
    ) -> None:
        self.graph = graph
        self.loss_fn = loss_fn
        self.params: List[Parameter] = self._collect_params()
        self.optimizer = optimizer_cls(self.params, lr=lr)

    # ------------------------------------------------------------------
    # Colección de parámetros
    # ------------------------------------------------------------------
    def _collect_params(self) -> List[Parameter]:
        params: List[Parameter] = []
        for block in self.graph.blocks.values():
            if isinstance(block, TRM_ACT_Block):
                params.extend([block.W_in, block.W_z, block.W_out, block.W_halt, block.b])
            elif isinstance(block, CognitiveBlock):
                perceiver = block.perceiver
                params.extend(getattr(perceiver, "input_weights", []))
                params.extend(getattr(perceiver, "memory_weights", []))
                params.extend(getattr(perceiver, "gate_weights", []))
                gate_bias = getattr(perceiver, "gate_bias", None)
                if gate_bias is not None:
                    params.append(gate_bias)
                bias = getattr(perceiver, "bias", None)
                if bias is not None:
                    params.append(bias)
                reasoner = block.reasoner
                params.extend(getattr(reasoner, "weights_in", []))
                params.extend(getattr(reasoner, "weights_mem", []))
                reasoner_bias = getattr(reasoner, "bias", None)
                if reasoner_bias is not None:
                    params.append(reasoner_bias)
                params.extend(getattr(block, "decision_weights", []))
                decision_bias = getattr(block, "decision_bias", None)
                if decision_bias is not None:
                    params.append(decision_bias)

        for proj in self.graph.projections.values():
            params.append(proj.W)
            params.append(proj.b)
        return params

    # ------------------------------------------------------------------
    # Entrenamiento global
    # ------------------------------------------------------------------
    def train_step(
        self,
        input_batch: Iterable[Dict[str, np.ndarray | List[float] | float]],
        target_batch: Iterable[np.ndarray | List[float] | float],
    ) -> float:
        total_loss = 0.0
        grads = [np.zeros_like(param.data, dtype=np.float32) for param in self.params]
        batch_inputs = list(input_batch)
        batch_targets = list(target_batch)
        contributions = 0

        for inputs, target in zip(batch_inputs, batch_targets):
            outputs = self.graph.forward(inputs)

            last_name = list(self.graph.blocks.keys())[-1]
            y_pred = outputs[last_name]
            y_true = _to_tensor(np.array(target).reshape(1, -1))

            # pérdida principal
            loss = self.loss_fn(y_pred, y_true)
            total_loss += float(loss.data)
            contributions += 1

            # Deep supervision: pérdidas locales por bloque
            for name, block in self.graph.blocks.items():
                output_tensor = outputs[name]
                if output_tensor.data.shape[1] == y_true.data.shape[1]:
                    local_loss = self.loss_fn(output_tensor, y_true)
                    total_loss += float(local_loss.data)
                    contributions += 1
                if isinstance(block, TRM_ACT_Block):
                    block_input = self.graph.last_inputs.get(name)
                    if block_input is not None:
                        deep_loss = block.deep_supervision_loss(block_input, y_true)
                        total_loss += float(deep_loss.data)
                        contributions += 1

            # gradiente simple (aproximado)
            diff = y_pred.data - y_true.data
            grad_scalar = np.mean(diff)
            for idx in range(len(grads)):
                grads[idx] += grad_scalar

        total_loss /= max(1, contributions)
        grads = [g / max(1, len(batch_inputs)) for g in grads]
        self.optimizer.step(grads)
        return total_loss

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def summary(self) -> None:
        print("=== GraphTrainer Summary ===")
        print(f"Blocks: {list(self.graph.blocks.keys())}")
        print(f"Parameters tracked: {len(self.params)}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print("================================")
