"""
Generadores de tareas para Curriculum Learning.

Cada tarea retorna (X, Y) donde:
- X: inputs (samples, features)
- Y: targets (samples, outputs)

Las tareas están ordenadas por dificultad creciente.
"""

import numpy as np
from typing import Tuple


def identity_task(n_features: int = 2, samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Identity: y = x (más simple).
    
    Aprende a copiar la entrada sin transformación.
    
    Args:
        n_features: Número de características de entrada
        samples: Número de muestras a generar
    
    Returns:
        X, Y con X.shape == Y.shape == (samples, n_features)
    """
    X = np.random.rand(samples, n_features).astype(np.float32)
    Y = X.copy()
    return X, Y


def xor_task(samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea XOR: y = x0 XOR x1 (dificultad baja).
    
    Problema clásico no linealmente separable.
    
    Args:
        samples: Número de muestras a generar
    
    Returns:
        X: (samples, 2), Y: (samples, 1)
    """
    X = np.random.randint(0, 2, size=(samples, 2)).astype(np.float32)
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32).reshape(-1, 1)
    return X, Y


def parity_task(n_bits: int = 3, samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Parity: y = (sum(x) % 2) (dificultad media).
    
    Calcula si la suma de bits es par o impar.
    
    Args:
        n_bits: Número de bits de entrada
        samples: Número de muestras a generar
    
    Returns:
        X: (samples, n_bits), Y: (samples, 1)
    """
    X = np.random.randint(0, 2, size=(samples, n_bits)).astype(np.float32)
    Y = (np.sum(X, axis=1) % 2).astype(np.float32).reshape(-1, 1)
    return X, Y


def sequence_task(length: int = 4, samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Sequence: Predecir el siguiente elemento (dificultad media-alta).
    
    Aprende a predecir el siguiente valor en una secuencia.
    
    Args:
        length: Longitud de la secuencia
        samples: Número de secuencias a generar
    
    Returns:
        X: (samples, length), Y: (samples, length) shifted
    """
    X = np.random.rand(samples, length).astype(np.float32)
    Y = np.roll(X, -1, axis=1)  # Shift left: predict next
    return X, Y


def reasoning_task(samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Reasoning: y = (x0 AND x1) OR (NOT x2) (dificultad alta).
    
    Requiere razonamiento lógico compuesto.
    
    Args:
        samples: Número de muestras a generar
    
    Returns:
        X: (samples, 3), Y: (samples, 1)
    """
    X = np.random.rand(samples, 3).astype(np.float32)
    # (x0 > 0.5 AND x1 > 0.5) OR (x2 < 0.5)
    Y = (((X[:, 0] > 0.5) & (X[:, 1] > 0.5)) | (X[:, 2] < 0.5)).astype(np.float32).reshape(-1, 1)
    return X, Y


def memory_task(sequence_length: int = 5, samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Memory: Recordar el primer elemento de la secuencia (dificultad alta).
    
    Requiere mantener información en memoria a través del tiempo.
    
    Args:
        sequence_length: Longitud de la secuencia
        samples: Número de secuencias a generar
    
    Returns:
        X: (samples, sequence_length), Y: (samples, 1) = primer elemento
    """
    X = np.random.rand(samples, sequence_length).astype(np.float32)
    Y = X[:, 0:1]  # Recordar el primer elemento
    return X, Y


def counting_task(max_value: int = 5, samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarea Counting: Contar cuántos 1s hay en la entrada (dificultad media).
    
    Args:
        max_value: Longitud máxima del vector
        samples: Número de muestras a generar
    
    Returns:
        X: (samples, max_value), Y: (samples, 1) = count of 1s
    """
    X = np.random.randint(0, 2, size=(samples, max_value)).astype(np.float32)
    Y = np.sum(X, axis=1, keepdims=True).astype(np.float32) / max_value  # Normalizado
    return X, Y


# Mapeo de nombres a funciones para fácil acceso
TASK_REGISTRY = {
    "identity": identity_task,
    "xor": xor_task,
    "parity": parity_task,
    "sequence": sequence_task,
    "reasoning": reasoning_task,
    "memory": memory_task,
    "counting": counting_task,
}


def get_task(name: str):
    """
    Obtiene una tarea por nombre.
    
    Args:
        name: Nombre de la tarea (identity, xor, parity, etc.)
    
    Returns:
        Función generadora de la tarea
    
    Raises:
        KeyError: Si la tarea no existe
    """
    if name not in TASK_REGISTRY:
        raise KeyError(f"Tarea '{name}' no encontrada. Disponibles: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]
