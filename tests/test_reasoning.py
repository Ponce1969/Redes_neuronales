"""Tests para el módulo de Cognitive Reasoning Layer."""

import numpy as np
import pytest

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.reasoning import (
    Reasoner,
    evaluate_reasoner,
    evolve_reasoner_on_task,
    extract_gates_history,
)


@pytest.fixture
def simple_graph():
    """Crea un grafo cognitivo simple para pruebas."""
    graph = CognitiveGraphHybrid()
    graph.add_block("input", CognitiveBlock(n_inputs=2, n_hidden=4, n_outputs=2))
    graph.add_block("hidden", CognitiveBlock(n_inputs=2, n_hidden=3, n_outputs=2))
    graph.add_block("output", CognitiveBlock(n_inputs=2, n_hidden=2, n_outputs=1))

    graph.connect("input", "hidden")
    graph.connect("hidden", "output")

    return graph


@pytest.fixture
def reasoner():
    """Crea un Reasoner para pruebas."""
    return Reasoner(n_inputs=12, n_hidden=16, n_blocks=3, seed=42)


def test_reasoner_initialization():
    """Test que el Reasoner se inicializa correctamente."""
    reasoner = Reasoner(n_inputs=10, n_hidden=20, n_blocks=5, seed=42)

    assert reasoner.n_inputs == 10
    assert reasoner.n_hidden == 20
    assert reasoner.n_blocks == 5
    assert reasoner.W1.shape == (10, 20)
    assert reasoner.W2.shape == (20, 5)


def test_reasoner_decide_softmax(reasoner):
    """Test que el Reasoner calcula gates en modo softmax."""
    z_list = [np.random.randn(1, 4).astype(np.float32) for _ in range(3)]
    gates = reasoner.decide(z_list, mode="softmax")

    assert len(gates) == 3
    assert all(isinstance(v, float) for v in gates.values())
    # Softmax debe sumar aproximadamente 1.0
    assert abs(sum(gates.values()) - 1.0) < 1e-5


def test_reasoner_decide_topk(reasoner):
    """Test que el Reasoner calcula gates en modo top-k."""
    z_list = [np.random.randn(1, 4).astype(np.float32) for _ in range(3)]
    gates = reasoner.decide(z_list, mode="topk", top_k=2)

    assert len(gates) == 3
    # Solo top_k bloques deben tener peso > 0
    active_blocks = sum(1 for v in gates.values() if v > 0)
    assert active_blocks == 2


def test_reasoner_decide_threshold(reasoner):
    """Test que el Reasoner calcula gates en modo threshold."""
    z_list = [np.random.randn(1, 4).astype(np.float32) for _ in range(3)]
    gates = reasoner.decide(z_list, mode="threshold")

    assert len(gates) == 3
    assert all(isinstance(v, float) for v in gates.values())


def test_reasoner_mutate(reasoner):
    """Test que la mutación genera un Reasoner diferente."""
    child = reasoner.mutate(scale=0.1, seed=123)

    assert child.n_inputs == reasoner.n_inputs
    assert child.n_hidden == reasoner.n_hidden
    assert child.n_blocks == reasoner.n_blocks

    # Los pesos deben ser diferentes
    assert not np.allclose(child.W1, reasoner.W1)
    assert not np.allclose(child.W2, reasoner.W2)


def test_reasoner_state_dict(reasoner):
    """Test serialización y deserialización del Reasoner."""
    state = reasoner.state_dict()

    assert "W1" in state
    assert "b1" in state
    assert "W2" in state
    assert "b2" in state

    # Crear nuevo reasoner y cargar estado
    new_reasoner = Reasoner(n_inputs=12, n_hidden=16, n_blocks=3)
    new_reasoner.load_state_dict(state)

    assert np.allclose(new_reasoner.W1, reasoner.W1)
    assert np.allclose(new_reasoner.W2, reasoner.W2)


def test_forward_with_reasoner(simple_graph, reasoner):
    """Test que forward_with_reasoner ejecuta correctamente."""
    inputs = {"input": [0.5, 0.5]}

    # Forward normal primero para que los bloques computen planes
    _ = simple_graph.forward(inputs)

    # Forward con reasoner
    outputs = simple_graph.forward_with_reasoner(inputs, reasoner, mode="softmax")

    assert len(outputs) == 3
    assert "output" in outputs
    assert hasattr(simple_graph, "last_gates")
    assert len(simple_graph.last_gates) == 3  # type: ignore[attr-defined]


def test_evaluate_reasoner(simple_graph, reasoner):
    """Test que evaluate_reasoner calcula loss correctamente."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([0, 1, 1, 0], dtype=np.float32)

    loss = evaluate_reasoner(simple_graph, reasoner, X, Y)

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_extract_gates_history(simple_graph, reasoner):
    """Test que extract_gates_history captura correctamente."""
    X = np.array([[0, 0], [0, 1]], dtype=np.float32)

    history = extract_gates_history(simple_graph, reasoner, X, mode="softmax")

    assert len(history) == 2
    assert all(len(gates) == 3 for gates in history)
    assert all("input" in gates for gates in history)


def test_evolve_reasoner_short(simple_graph):
    """Test que evolve_reasoner_on_task mejora el Reasoner (versión corta)."""
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    Y = np.array([0, 0], dtype=np.float32)

    initial_reasoner = Reasoner(n_inputs=12, n_hidden=16, n_blocks=3, seed=42)
    initial_loss = evaluate_reasoner(simple_graph, initial_reasoner, X, Y)

    # Solo 5 generaciones para test rápido
    evolved, history = evolve_reasoner_on_task(
        graph=simple_graph,
        base_reasoner=initial_reasoner,
        X=X,
        Y=Y,
        generations=5,
        pop_size=4,
        verbose=False,
    )

    assert len(history) == 6  # Initial + 5 generations
    # La evolución debe mantener o mejorar la loss
    assert history[-1] <= initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
