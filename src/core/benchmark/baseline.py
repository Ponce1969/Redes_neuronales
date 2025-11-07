"""
Estrategias Baseline - Puntos de referencia para comparación.

Define estrategias baseline para comparar contra el Reasoner entrenado.
"""

import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class BaselineStrategy:
    """
    Estrategia baseline para generar gates.
    
    Attributes:
        name: Nombre de la estrategia
        description: Descripción
        generator: Función que genera gates
    """
    name: str
    description: str
    generator: Callable[[int], np.ndarray]
    
    def generate_gates(self, n_blocks: int) -> np.ndarray:
        """
        Genera gates para N bloques.
        
        Args:
            n_blocks: Número de bloques
        
        Returns:
            Array de gates [0, 1]
        """
        return self.generator(n_blocks)


# ============================================================================
# Estrategias Baseline Pre-Definidas
# ============================================================================

def random_uniform_gates(n_blocks: int) -> np.ndarray:
    """
    Gates aleatorios uniformes en [0, 1].
    
    Baseline más básico: sin aprendizaje, puramente aleatorio.
    """
    return np.random.uniform(0, 1, size=n_blocks)


def random_softmax_gates(n_blocks: int) -> np.ndarray:
    """
    Gates aleatorios con distribución softmax.
    
    Similar a Reasoner con softmax pero sin aprendizaje.
    """
    logits = np.random.randn(n_blocks)
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()


def equal_gates(n_blocks: int) -> np.ndarray:
    """
    Todos los gates iguales (1/N).
    
    Baseline: activar todos los bloques por igual.
    """
    return np.ones(n_blocks) / n_blocks


def binary_random_gates(n_blocks: int) -> np.ndarray:
    """
    Gates binarios aleatorios (0 o 1).
    
    Cada bloque se activa o desactiva completamente.
    """
    return np.random.randint(0, 2, size=n_blocks).astype(float)


def topk_random_gates(n_blocks: int, k: int = 2) -> np.ndarray:
    """
    Activa K bloques aleatorios, resto a 0.
    
    Args:
        n_blocks: Total de bloques
        k: Cuántos activar
    """
    gates = np.zeros(n_blocks)
    selected = np.random.choice(n_blocks, size=min(k, n_blocks), replace=False)
    gates[selected] = 1.0 / len(selected)  # Normalizado
    return gates


def first_k_gates(n_blocks: int, k: int = 2) -> np.ndarray:
    """
    Siempre activa los primeros K bloques.
    
    Baseline: estrategia determinística simple.
    """
    gates = np.zeros(n_blocks)
    gates[:min(k, n_blocks)] = 1.0 / min(k, n_blocks)
    return gates


def last_k_gates(n_blocks: int, k: int = 2) -> np.ndarray:
    """
    Siempre activa los últimos K bloques.
    """
    gates = np.zeros(n_blocks)
    gates[-min(k, n_blocks):] = 1.0 / min(k, n_blocks)
    return gates


def gaussian_centered_gates(n_blocks: int, center: float = 0.5, std: float = 0.2) -> np.ndarray:
    """
    Gates con distribución gaussiana centrada.
    
    Bloques del medio tienen mayor activación.
    """
    positions = np.linspace(0, 1, n_blocks)
    gates = np.exp(-((positions - center) ** 2) / (2 * std ** 2))
    return gates / gates.sum()  # Normalizar


# ============================================================================
# Registry de Baselines
# ============================================================================

BASELINE_STRATEGIES = {
    "random_uniform": BaselineStrategy(
        name="random_uniform",
        description="Gates aleatorios uniformes [0, 1]",
        generator=random_uniform_gates,
    ),
    "random_softmax": BaselineStrategy(
        name="random_softmax",
        description="Gates aleatorios con softmax",
        generator=random_softmax_gates,
    ),
    "equal": BaselineStrategy(
        name="equal",
        description="Todos los gates iguales (1/N)",
        generator=equal_gates,
    ),
    "binary_random": BaselineStrategy(
        name="binary_random",
        description="Gates binarios aleatorios (0 o 1)",
        generator=binary_random_gates,
    ),
    "topk_random": BaselineStrategy(
        name="topk_random",
        description="Activa K bloques aleatorios",
        generator=lambda n: topk_random_gates(n, k=2),
    ),
    "first_k": BaselineStrategy(
        name="first_k",
        description="Siempre activa los primeros K bloques",
        generator=lambda n: first_k_gates(n, k=2),
    ),
    "last_k": BaselineStrategy(
        name="last_k",
        description="Siempre activa los últimos K bloques",
        generator=lambda n: last_k_gates(n, k=2),
    ),
    "gaussian": BaselineStrategy(
        name="gaussian",
        description="Distribución gaussiana centrada",
        generator=gaussian_centered_gates,
    ),
}


def get_baseline(name: str) -> BaselineStrategy:
    """
    Obtiene estrategia baseline por nombre.
    
    Args:
        name: Nombre de la baseline
    
    Returns:
        BaselineStrategy
    
    Raises:
        KeyError: Si no existe
    """
    if name not in BASELINE_STRATEGIES:
        available = list(BASELINE_STRATEGIES.keys())
        raise KeyError(
            f"Baseline '{name}' no encontrada. Disponibles: {available}"
        )
    
    return BASELINE_STRATEGIES[name]


def list_baselines() -> list:
    """Lista todas las baselines disponibles."""
    return list(BASELINE_STRATEGIES.keys())


# ============================================================================
# Baseline Reasoner (Mock)
# ============================================================================

class BaselineReasoner:
    """
    Reasoner baseline que usa una estrategia fija.
    
    No aprende, solo genera gates según estrategia.
    Útil para comparar contra Reasoner entrenado.
    """
    
    def __init__(self, strategy: str = "random_uniform", n_blocks: int = 3):
        """
        Inicializa baseline reasoner.
        
        Args:
            strategy: Nombre de la estrategia baseline
            n_blocks: Número de bloques
        """
        self.strategy_name = strategy
        self.baseline = get_baseline(strategy)
        self.n_blocks = n_blocks
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Genera gates (ignora el estado).
        
        Args:
            state: Estado del grafo (ignorado)
        
        Returns:
            Gates según estrategia baseline
        """
        return self.baseline.generate_gates(self.n_blocks)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Alias para predict."""
        return self.predict(state)
    
    def __repr__(self) -> str:
        return f"BaselineReasoner(strategy={self.strategy_name!r})"


# ============================================================================
# Evaluación de Baselines
# ============================================================================

def evaluate_baseline(
    baseline_strategy: str,
    graph,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[float, float]:
    """
    Evalúa una estrategia baseline en un task.
    
    Args:
        baseline_strategy: Nombre de la estrategia
        graph: CognitiveGraphHybrid
        X: Inputs
        Y: Targets
    
    Returns:
        (mse_loss, accuracy)
    """
    baseline = BaselineReasoner(baseline_strategy, n_blocks=len(graph.blocks))
    
    predictions = []
    
    for x in X:
        # Generar gates baseline
        gates = baseline.predict(x)
        
        # Forward con gates
        output = graph.forward_with_reasoner(x, gates)
        predictions.append(output)
    
    predictions = np.array(predictions)
    
    # Métricas
    mse_loss = np.mean((predictions - Y) ** 2)
    
    # Accuracy (si binario)
    if Y.shape[-1] == 1 or len(Y.shape) == 1:
        preds_binary = (predictions > 0.5).astype(float)
        accuracy = np.mean(preds_binary == Y)
    else:
        accuracy = 0.0
    
    return float(mse_loss), float(accuracy)
