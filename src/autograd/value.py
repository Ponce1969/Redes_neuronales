"""
Motor autograd Fase 6 - Nodo escalar con propagación automática.
Implementación didáctica sin dependencias externas.
"""

from __future__ import annotations
from typing import Callable, Set
import math


class Value:
    """
    Nodo escalar con soporte completo para autograd.
    Cada operación crea un nuevo Value con referencias a sus padres.
    """
    
    def __init__(self, data: float, _children: tuple['Value', ...] = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op = _op
        self.label = label  # Para depuración

    def __add__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    __radd__ = __add__

    def __sub__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    __rmul__ = __mul__

    def __truediv__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __pow__(self, exp: float) -> 'Value':
        assert isinstance(exp, (int, float))
        out = Value(self.data ** exp, (self,), f"**{exp}")
        
        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    # Activaciones
    def tanh(self) -> 'Value':
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), "relu")
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Propaga gradientes desde este nodo hacia atrás (propagación inversa)."""
        topo = []
        visited = set()
        
        def build(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        
        build(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op={self._op})"
