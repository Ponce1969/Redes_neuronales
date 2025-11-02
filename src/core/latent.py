"""
Módulo de espacio latente para Fase 5 - Variable latente z y proyector.
Proporciona sampling de z y proyección para planificación interna.
"""

from __future__ import annotations
import random
import math
from typing import List


def sample_gaussian_z(dim: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
    """Sample z ~ N(mu, sigma^2) (lista de floats)."""
    return [random.gauss(mu, sigma) for _ in range(dim)]


def sample_uniform_z(dim: int, low: float = -1.0, high: float = 1.0) -> List[float]:
    """Sample z ~ U(low, high)"""
    return [random.uniform(low, high) for _ in range(dim)]


def random_uniform(a: float, b: float) -> float:
    """Helper para generar números aleatorios uniformes."""
    return random.uniform(a, b)


class LatentProjector:
    """
    Proyector simple que transforma z (lista) a un vector de dimensión 'out_dim'.
    Implementado en Python puro.
    """

    def __init__(self, z_dim: int, out_dim: int, init_range: float = 0.5):
        self.z_dim = z_dim
        self.out_dim = out_dim
        # pesos: matriz out_dim x z_dim
        self.weights: List[List[float]] = [
            [random_uniform(-init_range, init_range) for _ in range(z_dim)]
            for _ in range(out_dim)
        ]
        self.bias: List[float] = [random_uniform(-init_range, init_range) for _ in range(out_dim)]

    def project(self, z: List[float]) -> List[float]:
        """
        Realiza una proyección lineal: out_i = tanh( sum_j w_ij * z_j + b_i )
        """
        assert len(z) == self.z_dim, f"z dimension mismatch: expected {self.z_dim}, got {len(z)}"
        out: List[float] = []
        for i in range(self.out_dim):
            s = self.bias[i]
            for j in range(self.z_dim):
                s += self.weights[i][j] * z[j]
            out.append(math.tanh(s))
        return out

    def __repr__(self) -> str:
        return f"<LatentProjector z_dim={self.z_dim}, out_dim={self.out_dim}>"


def sample_gaussian_z(dim: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
    """Sample z ~ N(mu, sigma^2) (lista de floats)."""
    return [random.gauss(mu, sigma) for _ in range(dim)]


def sample_uniform_z(dim: int, low: float = -1.0, high: float = 1.0) -> List[float]:
    """Sample z ~ U(low, high)"""
    return [random.uniform(low, high) for _ in range(dim)]


def reparametrize(mu: List[float], logvar: List[float]) -> List[float]:
    """
    Reparametrization trick: z = mu + eps * exp(0.5 * logvar)
    Útil para VAEs y entrenamiento con gradientes.
    """
    if len(mu) != len(logvar):
        raise ValueError("mu y logvar deben tener la misma dimensión")
    
    z = []
    for m, lv in zip(mu, logvar):
        eps = random.gauss(0.0, 1.0)
        z.append(m + eps * math.exp(0.5 * lv))
    return z


def clamp_z(z: List[float], min_val: float = -3.0, max_val: float = 3.0) -> List[float]:
    """Limita valores de z para evitar explosión de gradientes"""
    return [max(min(val, max_val), min_val) for val in z]


def z_distance(z1: List[float], z2: List[float]) -> float:
    """Distancia euclidiana entre dos vectores z"""
    if len(z1) != len(z2):
        raise ValueError("Los vectores z deben tener la misma dimensión")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2)))


class LatentSpace:
    """
    Gestor del espacio latente para RL curriculum.
    Permite tracking de exploración y diversidad.
    """
    
    def __init__(self, dim: int, strategy: str = "gaussian"):
        self.dim = dim
        self.strategy = strategy
        self.explored_zs: List[List[float]] = []
        self.z_history: List[Tuple[str, List[float], float]] = []  # (id, z, reward)
    
    def sample(self, strategy: str | None = None) -> List[float]:
        """Muestrea un nuevo vector z"""
        strategy = strategy or self.strategy
        
        if strategy == "gaussian":
            z = sample_gaussian_z(self.dim)
        elif strategy == "uniform":
            z = sample_uniform_z(self.dim)
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        
        return z
    
    def record(self, z: List[float], reward: float, task_id: str | None = None) -> None:
        """Registra un vector z con su recompensa"""
        task_id = task_id or str(uuid.uuid4())[:8]
        self.z_history.append((task_id, z.copy(), reward))
        self.explored_zs.append(z.copy())
    
    def get_diversity_reward(self, z: List[float], k: int = 5) -> float:
        """
        Calcula recompensa por diversidad basada en distancia a k vecinos más cercanos
        """
        if len(self.explored_zs) < k:
            return 1.0  # Recompensa máxima por exploración inicial
        
        distances = [z_distance(z, ez) for ez in self.explored_zs]
        distances.sort()
        avg_distance = sum(distances[:k]) / k
        
        # Normalizar a [0, 1]
        return min(avg_distance / 2.0, 1.0)
    
    def get_best_z(self, top_k: int = 1) -> List[List[float]]:
        """Obtiene los mejores vectores z basados en recompensa"""
        if not self.z_history:
            return []
        
        sorted_z = sorted(self.z_history, key=lambda x: x[2], reverse=True)
        return [z for _, z, _ in sorted_z[:top_k]]
    
    def clear_history(self) -> None:
        """Limpia el historial de exploración"""
        self.explored_zs.clear()
        self.z_history.clear()


def random_uniform(a: float, b: float) -> float:
    """Helper para generar números aleatorios uniformes."""
    return random.uniform(a, b)


class LatentProjector:
    """
    Proyector simple que transforma z (lista) a un vector de dimensión 'out_dim'.
    Implementado en Python puro; se puede sustituir luego por una versión vectorizada.
    """

    def __init__(self, z_dim: int, out_dim: int, init_range: float = 0.5):
        self.z_dim = z_dim
        self.out_dim = out_dim
        # pesos: matriz out_dim x z_dim (fila por salida)
        self.weights: List[List[float]] = [
            [random_uniform(-init_range, init_range) for _ in range(z_dim)]
            for _ in range(out_dim)
        ]
        self.bias: List[float] = [random_uniform(-init_range, init_range) for _ in range(out_dim)]

    def project(self, z: List[float]) -> List[float]:
        """
        Realiza una proyección lineal: out_i = tanh( sum_j w_ij * z_j + b_i )
        Devuelve vector de longitud out_dim.
        """
        assert len(z) == self.z_dim, "z dimension mismatch"
        out: List[float] = []
        for i in range(self.out_dim):
            s = self.bias[i]
            row = self.weights[i]
            for j in range(self.z_dim):
                s += row[j] * z[j]
            # activación no-lineal suave para estabilizar (tanh)
            out.append(math.tanh(s))
        return out

    def __repr__(self) -> str:
        return f"<LatentProjector z_dim={self.z_dim}, out_dim={self.out_dim}>"
