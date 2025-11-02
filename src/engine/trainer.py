"""
Trainer sencillo que entrena la red usando minibatches.
Soporta MSE y BCE (la función de pérdida y su derivada se pasan como parámetros).
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Tuple
from core.network import NeuralNetwork
from core import losses
from core.optimizers import Optimizer, SGD
import random
import math

Batch = List[Tuple[List[float], List[float]]]  # lista de (input, target)


class Trainer:
    def __init__(
        self,
        network: NeuralNetwork,
        loss_fn: Callable[[List[float], List[float]], float],
        loss_grad_fn: Callable[[List[float], List[float]], List[float]],
        optimizer: Optimizer | None = None,
        lr: float = 0.01,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        self.network = network
        self.loss_fn = loss_fn
        self.loss_grad_fn = loss_grad_fn
        self.optimizer = optimizer or SGD(lr=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _make_batches(self, dataset: Iterable[Tuple[List[float], List[float]]]) -> List[Batch]:
        data = list(dataset)
        if self.shuffle:
            random.shuffle(data)
        batches: List[Batch] = []
        for i in range(0, len(data), self.batch_size):
            batches.append(data[i : i + self.batch_size])
        return batches

    def train(
        self,
        dataset: Iterable[Tuple[List[float], List[float]]],
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        dataset: iterable of (input_vector, target_vector)
        """
        for epoch in range(1, epochs + 1):
            batches = self._make_batches(dataset)
            epoch_loss = 0.0
            for batch in batches:
                # For simplicity we run SGD per-sample inside batch (online)
                # Option: accumulate grads and average to apply mini-batch updates
                for x, y in batch:
                    # compute loss value for logging
                    y_pred = self.network.forward(x)
                    epoch_loss += self.loss_fn(y_pred, y)
                    # compute gradient (dL/dy) and do a train step per sample
                    dL_dy = self.loss_grad_fn(y_pred, y)
                    # We call train_step that expects loss_grad_fn signature (outputs, targets)
                    # For convenience, pass a lambda that returns the precomputed dL_dy
                    self.network.train_step(x, y, lambda out, tar: dL_dy, lr=self.lr)

                # after processing the batch we could do something (e.g., learning-rate schedule)

            epoch_loss /= max(1, len(list(dataset)))
            if verbose:
                print(f"[Epoch {epoch}/{epochs}] loss={epoch_loss:.6f}")

    def evaluate(self, dataset: Iterable[Tuple[List[float], List[float]]]) -> Tuple[float, int]:
        """
        Evalúa loss promedio y (para clasificación binaria) cuenta aciertos.
        Devuelve (avg_loss, correct_count)
        """
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataset:
            y_pred = self.network.forward(x)
            total_loss += self.loss_fn(y_pred, y)
            total += 1
            # simple binary accuracy if target scalar
            if len(y) == 1:
                pred_label = 1 if y_pred[0] >= 0.5 else 0
                if pred_label == int(y[0]):
                    correct += 1
        avg_loss = total_loss / max(1, total)
        return avg_loss, correct
