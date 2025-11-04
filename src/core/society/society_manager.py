"""Coordinador principal de la Sociedad Cognitiva."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from core.society.agent import CognitiveAgent
from core.society.communication import CommunicationChannel


@dataclass
class CycleStats:
    epoch: int
    losses: List[float]
    mean_loss: float


class SocietyManager:
    """Orquesta el entrenamiento colaborativo entre múltiples agentes."""

    def __init__(self, agents: Iterable[CognitiveAgent]) -> None:
        self.agents: List[CognitiveAgent] = list(agents)
        self.channel = CommunicationChannel(self.agents)
        self.history: List[CycleStats] = []

    def run_cycle(
        self,
        X: Sequence[np.ndarray],
        Y: Sequence[np.ndarray],
        epochs: int = 50,
        exchange_every: int = 10,
        broadcast_top: bool = False,
    ) -> None:
        """Ejecuta un ciclo colaborativo de entrenamiento."""

        if not self.agents:
            return

        for epoch in range(1, epochs + 1):
            losses = []
            for agent in self.agents:
                loss = agent.train_once(X, Y)
                losses.append(loss)

            stats = CycleStats(epoch=epoch, losses=losses, mean_loss=float(np.mean(losses)))
            self.history.append(stats)

            if exchange_every and epoch % exchange_every == 0:
                self.channel.exchange_memories()
                print(f"[Society] intercambio de memorias en epoch {epoch}")

            if broadcast_top and exchange_every and epoch % exchange_every == 0:
                self.channel.broadcast_top_experiences()
                print(f"[Society] broadcast de mejores experiencias en epoch {epoch}")

    def top_agent(self) -> CognitiveAgent | None:
        """Devuelve el agente con mejor rendimiento."""

        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.performance)
