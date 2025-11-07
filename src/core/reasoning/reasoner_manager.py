"""
ReasonerManager: Controlador centralizado del Reasoner para API y dashboard.

Gestiona el ciclo de vida del Reasoner incluyendo:
- Decisiones de gating en tiempo real
- Evolución en background (thread-safe)
- Persistencia de estado
- Historial de gates para visualización
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from src.core.reasoning.reasoner import Reasoner  # type: ignore
except ModuleNotFoundError:
    from core.reasoning.reasoner import Reasoner  # type: ignore


class ReasonerManager:
    """
    Controlador centralizado del Reasoner con capacidades de:
    - Gestión thread-safe de decisiones
    - Evolución asíncrona en background
    - Persistencia automática (.npz)
    - Historial de gates para análisis
    """

    def __init__(
        self, n_inputs: int, n_hidden: int, n_blocks: int, seed: Optional[int] = None
    ) -> None:
        """
        Inicializa el ReasonerManager.
        
        Args:
            n_inputs: Dimensión de entrada (concatenación de z_plan)
            n_hidden: Neuronas en capa oculta del Reasoner
            n_blocks: Número de bloques en el grafo cognitivo
            seed: Semilla para reproducibilidad
        """
        self.reasoner = Reasoner(n_inputs, n_hidden, n_blocks, seed=seed)
        self.lock = threading.RLock()

        # Estado de evolución
        self.running = False
        self.best_loss = np.inf
        self.generation = 0
        self.total_generations = 0

        # Historial de gates (últimos 100)
        self.gates_history: List[Dict[int, float]] = []
        self.max_history = 100

        # Estadísticas
        self.predict_calls = 0
        self.evolution_runs = 0

    # ========================================================================
    # DECISIONES DE GATING
    # ========================================================================

    def decide(
        self, z_per_block: List[np.ndarray], mode: str = "softmax", **kwargs: Any
    ) -> Dict[int, float]:
        """
        Calcula gates para cada bloque del grafo (thread-safe).
        
        Args:
            z_per_block: Lista de vectores latentes (uno por bloque)
            mode: Modo de gating ('softmax', 'topk', 'threshold')
            **kwargs: Argumentos adicionales (temp, top_k, etc.)
            
        Returns:
            Diccionario {block_index: gate_weight}
        """
        with self.lock:
            gates = self.reasoner.decide(z_per_block, mode=mode, **kwargs)
            
            # Registrar en historial
            self.gates_history.append(gates.copy())
            if len(self.gates_history) > self.max_history:
                self.gates_history.pop(0)
            
            self.predict_calls += 1
            
            return gates

    def get_recent_gates(self, n: int = 10) -> List[Dict[int, float]]:
        """Obtiene los últimos N gates calculados."""
        with self.lock:
            return self.gates_history[-n:]

    # ========================================================================
    # EVOLUCIÓN ASÍNCRONA
    # ========================================================================

    def evolve_async(
        self,
        evaluate_fn: Callable[[Reasoner], float],
        generations: int = 50,
        pop_size: int = 8,
        mutation_scale: float = 0.03,
    ) -> bool:
        """
        Lanza evolución del Reasoner en background.
        
        Args:
            evaluate_fn: Función que evalúa un Reasoner y devuelve loss
            generations: Número de generaciones a evolucionar
            pop_size: Tamaño de población por generación
            mutation_scale: Magnitud de mutación gaussiana
            
        Returns:
            True si la evolución se inició, False si ya está corriendo
        """
        if self.running:
            return False

        def worker() -> None:
            """Worker thread que ejecuta la evolución."""
            with self.lock:
                self.running = True
                self.generation = 0
                self.total_generations = generations
                self.evolution_runs += 1

            try:
                for gen in range(generations):
                    # Generar mutantes
                    children = []
                    with self.lock:
                        for _ in range(pop_size):
                            child = self.reasoner.mutate(scale=mutation_scale)
                            children.append(child)

                    # Evaluar fuera del lock (evaluación puede ser costosa)
                    best_child = None
                    best_child_loss = self.best_loss

                    for child in children:
                        loss = evaluate_fn(child)
                        if loss < best_child_loss:
                            best_child = child
                            best_child_loss = loss

                    # Actualizar si mejoró
                    with self.lock:
                        if best_child is not None and best_child_loss < self.best_loss:
                            self.reasoner = best_child
                            self.best_loss = best_child_loss

                        self.generation = gen + 1

                    # Pequeña pausa para no saturar CPU
                    time.sleep(0.05)

            finally:
                with self.lock:
                    self.running = False

        # Lanzar thread daemon
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        return True

    def stop_evolution(self) -> bool:
        """
        Detiene la evolución en curso (marca flag, thread terminará naturalmente).
        
        Returns:
            True si había evolución corriendo
        """
        with self.lock:
            if self.running:
                self.total_generations = self.generation  # Forzar fin
                return True
            return False

    # ========================================================================
    # PERSISTENCIA
    # ========================================================================

    def save(self, path: str) -> bool:
        """
        Guarda el estado del Reasoner en formato .npz comprimido.
        
        Args:
            path: Ruta del archivo (se añade .npz si no lo tiene)
            
        Returns:
            True si guardó exitosamente
        """
        try:
            with self.lock:
                state_dict = self.reasoner.state_dict()
                
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
                
                # Añadir .npz si no lo tiene
                if not path.endswith(".npz"):
                    path = f"{path}.npz"
                
                # Guardar con compresión
                np.savez_compressed(path, **state_dict)
                
            return True
        except Exception as e:
            print(f"[ReasonerManager] Error al guardar: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Carga el estado del Reasoner desde .npz.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            True si cargó exitosamente
        """
        try:
            # Añadir .npz si no lo tiene
            if not path.endswith(".npz"):
                path = f"{path}.npz"
            
            if not os.path.exists(path):
                print(f"[ReasonerManager] Archivo no encontrado: {path}")
                return False

            with self.lock:
                loaded = np.load(path)
                state_dict = {k: loaded[k] for k in loaded.files}
                self.reasoner.load_state_dict(state_dict)
                
            print(f"[ReasonerManager] Estado cargado desde {path}")
            return True
            
        except Exception as e:
            print(f"[ReasonerManager] Error al cargar: {e}")
            return False

    # ========================================================================
    # ESTADO Y ESTADÍSTICAS
    # ========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Devuelve el estado actual del ReasonerManager.
        
        Returns:
            Diccionario con métricas y estado
        """
        with self.lock:
            # Convertir np.inf a un valor manejable para JSON
            loss_value = float(self.best_loss)
            if not np.isfinite(loss_value):
                loss_value = 1.0  # Valor por defecto si es inf o nan
            
            return {
                "running": self.running,
                "generation": self.generation,
                "total_generations": self.total_generations,
                "best_loss": loss_value,
                "predict_calls": self.predict_calls,
                "evolution_runs": self.evolution_runs,
                "history_length": len(self.gates_history),
                "progress": (
                    (self.generation / self.total_generations * 100)
                    if self.total_generations > 0
                    else 0.0
                ),
            }

    def reset_stats(self) -> None:
        """Reinicia estadísticas (útil para testing)."""
        with self.lock:
            self.predict_calls = 0
            self.evolution_runs = 0
            self.gates_history.clear()
