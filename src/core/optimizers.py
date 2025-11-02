"""
Optimizadores modulares: SGD, SGD + Momentum, Adam.
Permite experimentar con distintos métodos sin modificar la red.
"""

from __future__ import annotations
from typing import List, Dict
import math


class Optimizer:
    """Clase base para optimizadores."""

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        raise NotImplementedError


class SGD(Optimizer):
    """Descenso de gradiente estocástico simple."""
    
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        return [p - self.lr * g for p, g in zip(params, grads)]


class SGDMomentum(Optimizer):
    """SGD con momentum para acelerar convergencia."""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Dict[int, float] = {}

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            v_prev = self.velocity.get(i, 0.0)
            v = self.momentum * v_prev - self.lr * g
            self.velocity[i] = v
            new_params.append(p + v)
        return new_params


class Adam(Optimizer):
    """Adam optimizer con momentum adaptativo y learning rate adaptativo."""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, float] = {}
        self.v: Dict[int, float] = {}
        self.t = 0

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        new_params = []
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m_prev = self.m.get(i, 0.0)
            v_prev = self.v.get(i, 0.0)
            
            # Actualizar momentos
            m = self.beta1 * m_prev + (1 - self.beta1) * g
            v = self.beta2 * v_prev + (1 - self.beta2) * (g ** 2)
            
            # Corrección de sesgo
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Actualizar parámetros
            p_new = p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            new_params.append(p_new)
            
            # Guardar estado
            self.m[i] = m
            self.v[i] = v
        return new_params


class RMSprop(Optimizer):
    """RMSprop optimizer para learning rate adaptativo."""
    
    def __init__(self, lr: float = 0.001, rho: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.cache: Dict[int, float] = {}

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache_prev = self.cache.get(i, 0.0)
            cache = self.rho * cache_prev + (1 - self.rho) * (g ** 2)
            self.cache[i] = cache
            p_new = p - self.lr * g / (math.sqrt(cache) + self.eps)
            new_params.append(p_new)
        return new_params