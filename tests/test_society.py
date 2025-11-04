from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Iterable, List

from core.society.communication import CommunicationChannel
from core.society.society_manager import SocietyManager


class StubMemory:
    def __init__(self, episodes: List[dict[str, Any]]) -> None:
        self.buffer = list(episodes)

    def sample(self, n: int) -> List[dict[str, Any]]:  # pragma: no cover - trivial
        return self.buffer[:n]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


class StubAgent:
    def __init__(self, name: str, episodes: Iterable[dict[str, Any]] = ()) -> None:
        self.name = name
        self.received: List[dict[str, Any]] = []
        self.performance = 0.0
        self.memory_system = SimpleNamespace(memory=StubMemory(list(episodes)))

    def receive_experiences(self, episodes: Iterable[dict[str, Any]]) -> None:
        self.received.extend(episodes)

    def share_top_experiences(self, top_k: int = 4) -> List[dict[str, Any]]:  # pragma: no cover - unused
        return []

    def train_once(self, X, Y) -> float:  # pragma: no cover - unused
        return 0.0


def test_exchange_memories_transfers_episodes() -> None:
    random.seed(42)
    episode = {"input": {"input": [0, 1]}, "target": [1], "output": {"decision": [0.9]}, "loss": 0.1, "attention": {}}

    source = StubAgent("source", [episode])
    target = StubAgent("target")

    channel = CommunicationChannel([source, target])
    channel.exchange_memories(n_samples=1)

    assert target.received, "El agente receptor debería haber recibido episodios"
    assert target.received[0]["loss"] == 0.1


def test_society_manager_top_agent_returns_highest_performance() -> None:
    a1 = StubAgent("a1")
    a2 = StubAgent("a2")
    a3 = StubAgent("a3")

    a1.performance = -0.5
    a2.performance = -0.2
    a3.performance = -0.8

    manager = SocietyManager([a1, a2, a3])

    assert manager.top_agent() is a2
