"""Ciclo autónomo para coordinar la sociedad cognitiva."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from core.federation.federated_client import FederatedClient
from core.persistence.persistence_manager import PersistenceManager
from core.scheduler.scheduler_config import SchedulerConfig
from core.society.society_manager import SocietyManager


@dataclass(slots=True)
class _TimerEntry:
    interval: float
    enabled: bool
    runner: Callable[[], None]


class CognitiveScheduler:
    """Coordina entrenamiento, persistencia, federación y consolidación."""

    def __init__(
        self,
        society: SocietyManager,
        config: SchedulerConfig,
        persistence: Optional[PersistenceManager] = None,
        federated_client: Optional[FederatedClient] = None,
        monitor: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.society = society
        self.config = config
        self.persistence = persistence or PersistenceManager(society)
        self.federated_client = federated_client
        self.monitor_callback = monitor

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.RLock()

    def start(self) -> None:
        """Lanza el loop autónomo en un hilo daemon."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running.set()
            self._thread = threading.Thread(target=self._loop, name="CognitiveScheduler", daemon=True)
            self._thread.start()
            print("[Scheduler] Sistema cognitivo autónomo iniciado ✅")

    def stop(self, join: bool = False) -> None:
        """Detiene el loop y, opcionalmente, espera al hilo."""

        with self._lock:
            self._running.clear()
            thread = self._thread
        if join and thread is not None:
            thread.join()

    # ------------------------------------------------------------------
    # Ciclos individuales
    # ------------------------------------------------------------------

    def _train_cycle(self) -> None:
        dataset_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        dataset_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        for agent in self.society.agents:
            loss = agent.train_once(dataset_x, dataset_y)
            graph_monitor = getattr(agent.graph, "monitor", None)
            if graph_monitor is not None:
                graph_monitor.track_loss(loss)
            if self.monitor_callback is not None:
                self.monitor_callback(loss)

        print("[Scheduler] Entrenamiento colaborativo completado")

    def _save_cycle(self) -> None:
        self.persistence.save_all()

    def _federation_cycle(self) -> None:
        client = self.federated_client
        if client is None:
            return

        for agent in self.society.agents:
            try:
                client.agent = agent
                client.send_weights()
            except Exception as exc:  # pragma: no cover - logging side effect
                print(f"[Scheduler] Error enviando pesos de {agent.name}: {exc}")

        for agent in self.society.agents:
            try:
                client.agent = agent
                client.receive_global_weights()
            except Exception as exc:  # pragma: no cover - logging side effect
                print(f"[Scheduler] Error recibiendo pesos para {agent.name}: {exc}")

    def _evolution_cycle(self) -> None:
        self.society.channel.exchange_memories()
        print("[Scheduler] Intercambio de memorias completado")

    def _sleep_cycle(self) -> None:
        for agent in self.society.agents:
            memory_system = getattr(agent, "memory_system", None)
            if memory_system is not None:
                try:
                    memory_system.sleep_and_replay()
                except Exception as exc:  # pragma: no cover
                    print(f"[Scheduler] Error en sleep de {agent.name}: {exc}")
        print("[Scheduler] Fase de sueño completada")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        now = time.time()
        timers: Dict[str, float] = {
            "train": now,
            "save": now,
            "federation": now,
            "evolution": now,
            "sleep": now,
        }

        schedule = {
            "train": _TimerEntry(self.config.train_interval, True, self._train_cycle),
            "save": _TimerEntry(self.config.save_interval, True, self._save_cycle),
            "federation": _TimerEntry(
                self.config.federation_interval,
                self.config.enable_federation,
                self._federation_cycle,
            ),
            "evolution": _TimerEntry(
                self.config.evolution_interval,
                self.config.enable_evolution,
                self._evolution_cycle,
            ),
            "sleep": _TimerEntry(
                self.config.sleep_interval,
                self.config.enable_sleep,
                self._sleep_cycle,
            ),
        }

        while self._running.is_set():
            now = time.time()
            for key, entry in schedule.items():
                if not entry.enabled or entry.interval <= 0:
                    continue
                if now - timers[key] >= entry.interval:
                    self._run_cycle(entry.runner, key)
                    timers[key] = now

            time.sleep(max(0.1, self.config.loop_sleep))

    def _run_cycle(self, func: Callable[[], None], name: str) -> None:
        try:
            func()
        except Exception as exc:  # pragma: no cover
            print(f"[Scheduler] Error en ciclo '{name}': {exc}")


__all__ = ["CognitiveScheduler"]
