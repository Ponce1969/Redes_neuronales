"""
Tests unitarios para el sistema de Curriculum Learning.

Verifica el funcionamiento de tasks, stages, evaluator y manager.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.curriculum import (
    tasks,
    CognitiveMetrics,
    CurriculumStage,
    CurriculumManager,
    CurriculumEvaluator,
    create_standard_curriculum,
)
from core.reasoning.reasoner import Reasoner
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock


# ============================================================================
# Tests de Tasks
# ============================================================================

def test_identity_task():
    """Test de la tarea identity."""
    X, Y = tasks.identity_task(n_features=3, samples=10)
    
    assert X.shape == (10, 3)
    assert Y.shape == (10, 3)
    assert np.allclose(X, Y), "Identity task debe retornar Y == X"


def test_xor_task():
    """Test de la tarea XOR."""
    X, Y = tasks.xor_task(samples=8)
    
    assert X.shape == (8, 2)
    assert Y.shape == (8, 1)
    assert np.all((Y == 0) | (Y == 1)), "XOR output debe ser binario"
    
    # Verificar lógica XOR
    for x, y in zip(X, Y):
        expected = int(x[0]) ^ int(x[1])
        assert y[0] == expected


def test_parity_task():
    """Test de la tarea parity."""
    X, Y = tasks.parity_task(n_bits=4, samples=16)
    
    assert X.shape == (16, 4)
    assert Y.shape == (16, 1)
    assert np.all((Y == 0) | (Y == 1)), "Parity output debe ser binario"


def test_task_registry():
    """Test del registro de tareas."""
    assert "xor" in tasks.TASK_REGISTRY
    assert "identity" in tasks.TASK_REGISTRY
    
    xor_func = tasks.get_task("xor")
    X, Y = xor_func()
    assert X.shape[0] > 0  # Debe generar samples


def test_invalid_task():
    """Test de tarea inválida."""
    with pytest.raises(KeyError):
        tasks.get_task("nonexistent_task")


# ============================================================================
# Tests de Metrics
# ============================================================================

def test_mse_loss():
    """Test de MSE loss."""
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 3.0])
    
    loss = CognitiveMetrics.mse_loss(preds, targets)
    assert loss == 0.0, "MSE debe ser 0 para predicciones perfectas"


def test_accuracy():
    """Test de accuracy."""
    preds = np.array([0.9, 0.1, 0.8, 0.2])
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    
    acc = CognitiveMetrics.accuracy(preds, targets, threshold=0.5)
    assert acc == 1.0, "Accuracy debe ser 1.0 para predicciones correctas"


def test_gate_metrics():
    """Test de métricas de gates."""
    gates_history = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.4, 0.4, 0.2]),
        np.array([0.6, 0.2, 0.2]),
    ]
    
    metrics = CognitiveMetrics.gate_metrics(gates_history)
    
    assert 'gate_diversity' in metrics
    assert 'gate_entropy' in metrics
    assert 'gate_utilization' in metrics
    assert 0 <= metrics['gate_diversity'] <= 1
    assert metrics['gate_entropy'] >= 0


def test_compute_all_metrics():
    """Test de cómputo de todas las métricas."""
    preds = np.random.rand(10, 1)
    targets = np.random.rand(10, 1)
    
    metrics = CognitiveMetrics.compute_all(preds, targets)
    
    assert 'mse_loss' in metrics
    assert 'mae_loss' in metrics
    assert 'stability' in metrics


# ============================================================================
# Tests de CurriculumStage
# ============================================================================

def test_curriculum_stage_creation():
    """Test de creación de etapa."""
    stage = CurriculumStage(
        name="test_stage",
        task_generator=tasks.xor_task,
        difficulty=2,
        max_epochs=50,
        success_threshold=0.05,
    )
    
    assert stage.name == "test_stage"
    assert stage.difficulty == 2
    assert not stage.completed


def test_stage_mark_completed():
    """Test de marcar etapa como completada."""
    stage = CurriculumStage(
        name="test",
        task_generator=tasks.identity_task,
        difficulty=1,
    )
    
    metrics = {'mse_loss': 0.01, 'accuracy': 0.95}
    stage.mark_completed(epochs=25, metrics=metrics)
    
    assert stage.completed
    assert stage.epochs_to_complete == 25
    assert stage.best_metrics['mse_loss'] == 0.01


def test_stage_validation():
    """Test de validación de parámetros."""
    # Difficulty inválido
    with pytest.raises(ValueError):
        CurriculumStage(
            name="bad",
            task_generator=tasks.xor_task,
            difficulty=15,  # > 10
        )
    
    # Thresholds inválidos
    with pytest.raises(ValueError):
        CurriculumStage(
            name="bad",
            task_generator=tasks.xor_task,
            success_threshold=0.5,
            fail_threshold=0.3,  # < success_threshold
        )


def test_create_standard_curriculum():
    """Test de creación del curriculum estándar."""
    stages = create_standard_curriculum()
    
    assert len(stages) >= 5
    assert all(isinstance(s, CurriculumStage) for s in stages)
    
    # Verificar que están ordenadas por dificultad
    difficulties = [s.difficulty for s in stages]
    assert difficulties == sorted(difficulties)


# ============================================================================
# Tests de Evaluator
# ============================================================================

@pytest.fixture
def simple_graph():
    """Fixture de grafo simple."""
    graph = CognitiveGraphHybrid()
    
    block = CognitiveBlock(input_dim=4, hidden_dim=8, name="test_block")
    graph.add_block("sensor", block)
    
    return graph


@pytest.fixture
def simple_reasoner():
    """Fixture de reasoner simple."""
    return Reasoner(n_inputs=16, n_hidden=16, n_blocks=1)


def test_evaluator_creation(simple_graph):
    """Test de creación del evaluator."""
    evaluator = CurriculumEvaluator(simple_graph)
    
    assert evaluator.graph is simple_graph
    assert evaluator.mode == "softmax"


def test_evaluator_single(simple_graph, simple_reasoner):
    """Test de evaluación de una muestra."""
    evaluator = CurriculumEvaluator(simple_graph)
    
    x = np.random.rand(4)
    y = np.random.rand(8)
    
    loss = evaluator.evaluate_single(simple_reasoner, x, y)
    
    assert isinstance(loss, float)
    assert loss >= 0


# ============================================================================
# Tests de CurriculumManager (básicos)
# ============================================================================

def test_manager_creation(simple_graph):
    """Test de creación del manager."""
    from core.reasoning.reasoner_manager import ReasonerManager
    
    reasoner_mgr = ReasonerManager(n_inputs=16, n_hidden=16, n_blocks=1)
    
    manager = CurriculumManager(
        reasoner_manager=reasoner_mgr,
        graph=simple_graph,
        auto_save=False,
    )
    
    assert manager.reasoner_manager is reasoner_mgr
    assert manager.graph is simple_graph
    assert len(manager.stages) == 0


def test_manager_add_stage(simple_graph):
    """Test de añadir etapas al manager."""
    from core.reasoning.reasoner_manager import ReasonerManager
    
    reasoner_mgr = ReasonerManager(n_inputs=16, n_hidden=16, n_blocks=1)
    manager = CurriculumManager(reasoner_mgr, simple_graph, auto_save=False)
    
    stage = CurriculumStage(
        name="test",
        task_generator=tasks.identity_task,
        difficulty=1,
    )
    
    manager.add_stage(stage)
    
    assert len(manager.stages) == 1
    assert manager.stages[0].name == "test"


def test_manager_status(simple_graph):
    """Test de obtener estado del manager."""
    from core.reasoning.reasoner_manager import ReasonerManager
    
    reasoner_mgr = ReasonerManager(n_inputs=16, n_hidden=16, n_blocks=1)
    manager = CurriculumManager(reasoner_mgr, simple_graph, auto_save=False)
    
    status = manager.status()
    
    assert 'running' in status
    assert 'current_stage_idx' in status
    assert 'total_stages' in status
    assert status['running'] is False


def test_manager_reset(simple_graph):
    """Test de reset del manager."""
    from core.reasoning.reasoner_manager import ReasonerManager
    
    reasoner_mgr = ReasonerManager(n_inputs=16, n_hidden=16, n_blocks=1)
    manager = CurriculumManager(reasoner_mgr, simple_graph, auto_save=False)
    
    # Simular progreso
    manager.current_stage_idx = 2
    manager.history = [{'stage': 'test', 'loss': 0.1}]
    
    manager.reset()
    
    assert manager.current_stage_idx == 0
    assert len(manager.history) == 0


# ============================================================================
# Tests de integración
# ============================================================================

def test_end_to_end_mini_curriculum(simple_graph):
    """Test de integración: ejecutar mini-curriculum."""
    from core.reasoning.reasoner_manager import ReasonerManager
    
    reasoner_mgr = ReasonerManager(n_inputs=16, n_hidden=16, n_blocks=1)
    manager = CurriculumManager(reasoner_mgr, simple_graph, auto_save=False)
    
    # Añadir una etapa muy simple con pocos epochs
    stage = CurriculumStage(
        name="identity",
        task_generator=lambda: tasks.identity_task(n_features=2, samples=4),
        difficulty=1,
        max_epochs=5,  # Muy pocos para test rápido
        success_threshold=0.5,  # Threshold alto para pasar rápido
        fail_threshold=1.0,
        log_interval=1,
    )
    
    manager.add_stage(stage)
    
    # Ejecutar (debería terminar rápido)
    history = manager.run()
    
    assert len(history) > 0
    assert not manager.running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
