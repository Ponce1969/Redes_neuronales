"""Canales de comunicación entre agentes cognitivos."""

from __future__ import annotations

import random
from typing import Iterable, List

from core.society.agent import CognitiveAgent


class CommunicationChannel:
    """Intercambia conocimiento entre agentes de la sociedad."""

    def __init__(self, agents: Iterable[CognitiveAgent]) -> None:
        self.agents: List[CognitiveAgent] = list(agents)

    def exchange_memories(self, n_samples: int = 4) -> None:
        """Comparte experiencias entre pares aleatorios."""

        if not self.agents:
            return

        for agent in self.agents:
            peers = [peer for peer in self.agents if peer is not agent]
            if not peers:
                continue

            peer = random.choice(peers)
            if len(peer.memory_system.memory) == 0:
                continue

            samples = peer.memory_system.memory.sample(n_samples)
            agent.receive_experiences(samples)

    def broadcast_top_experiences(self, top_k: int = 3) -> None:
        """Cada agente comparte sus mejores episodios con todos los demás."""

        if not self.agents:
            return

        for agent in self.agents:
            top_experiences = agent.share_top_experiences(top_k)
            for peer in self.agents:
                if peer is agent:
                    continue
                peer.receive_experiences(top_experiences)
