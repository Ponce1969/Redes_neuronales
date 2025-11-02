"""
Tests de estabilidad y memoria compartida del CognitiveGraph.
Verifica:
 - Flujo correcto entre bloques.
 - Persistencia y actualización coherente de memoria.
 - Estabilidad ante variaciones pequeñas de entrada.
"""

from __future__ import annotations

import os
import random
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from core.cognitive_block import CognitiveBlock
from core.cognitive_graph import CognitiveGraph
from autograd.value import Value


def build_test_graph() -> CognitiveGraph:
    """Crea un grafo cognitivo mínimo (percepción→razonamiento→decisión)."""
    state = random.getstate()
    random.seed(42)
    try:
        graph = CognitiveGraph()

        block_p = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)
        block_r = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)
        block_d = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)

        graph.add_block("perception", block_p)
        graph.add_block("reasoning", block_r)
        graph.add_block("decision", block_d)

        graph.connect("perception", "reasoning")
        graph.connect("reasoning", "decision")
    finally:
        random.setstate(state)

    return graph


def test_graph_basic_flow() -> None:
    """El grafo debe propagar señales correctamente entre bloques conectados."""
    graph = build_test_graph()
    inputs = {"perception": [0.5]}
    outputs = graph.step(inputs)

    assert "decision" in outputs, "Bloque de decisión no produjo salida."
    assert isinstance(outputs["decision"][0], float)
    assert -1.0 <= outputs["decision"][0] <= 1.0, "Salida fuera de rango esperable."


def test_memory_shared_consistency() -> None:
    """La memoria compartida debe contener una copia válida de cada bloque."""
    graph = build_test_graph()
    graph.step({"perception": [0.3]})
    shared = graph.shared_memory

    assert "perception" in shared and "reasoning" in shared, "Memoria compartida incompleta."
    assert all(isinstance(v, Value) for v in shared["perception"]), "Estados de memoria inválidos."
    assert len(shared["perception"]) == len(graph.blocks["perception"].perceiver.memory.state)


def test_stability_small_input_variation() -> None:
    """
    Si dos entradas son cercanas, las salidas deben ser parecidas.
    Evalúa estabilidad local de las transformaciones cognitivas.
    """
    graph = build_test_graph()
    out1 = graph.step({"perception": [0.40]})
    out2 = graph.step({"perception": [0.42]})

    diff = abs(out1["decision"][0] - out2["decision"][0])
    assert diff < 0.2, f"Salida inestable: diferencia {diff:.4f} demasiado alta."


def test_connection_integrity() -> None:
    """Verifica que las conexiones entre bloques se mantengan intactas."""
    graph = build_test_graph()
    conns = graph.connections

    assert "reasoning" in conns and "perception" in conns["reasoning"], (
        "Conexión percepción→razonamiento no establecida."
    )

    assert "decision" in conns and "reasoning" in conns["decision"], (
        "Conexión razonamiento→decisión no establecida."
    )


def test_repeated_forward_consistency() -> None:
    """
    Ejecutar varios pasos consecutivos no debe romper el grafo.
    Verifica estabilidad de estados en secuencias.
    """
    graph = build_test_graph()
    last_output = None
    for i in range(5):
        outputs = graph.step({"perception": [0.2 + i * 0.1]})
        assert isinstance(outputs, dict)
        assert "decision" in outputs
        last_output = outputs["decision"][0]

    # La última salida debe ser un float válido (no NaN, no inf)
    assert isinstance(last_output, float)
    assert abs(last_output) < 10.0, "Salida divergente en ejecución prolongada."
