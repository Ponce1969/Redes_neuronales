"""
Tests unitarios para el sistema de Benchmarking.

Verifica configuraciones, métricas, comparación estadística y reproducibilidad.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.benchmark import (
    BenchmarkConfig,
    BenchmarkMetrics,
    AggregatedMetrics,
    BenchmarkProvenance,
    BaselineReasoner,
    BenchmarkComparator,
    ComparisonResult,
    BENCHMARK_CONFIGS,
    BASELINE_STRATEGIES,
    get_config,
    list_configs,
    get_baseline,
    list_baselines,
    create_custom_config,
    calculate_loss_stability,
    calculate_loss_trend,
    calculate_gate_consistency,
    find_convergence_epoch,
)


# ============================================================================
# Tests de BenchmarkConfig
# ============================================================================

def test_benchmark_config_creation():
    """Test de creación de config."""
    config = BenchmarkConfig(
        name="test_config",
        description="Test configuration",
        seed=42,
        n_runs=5,
    )
    
    assert config.name == "test_config"
    assert config.seed == 42
    assert config.n_runs == 5


def test_config_hash():
    """Test de hash único."""
    config1 = BenchmarkConfig(name="test1", seed=42)
    config2 = BenchmarkConfig(name="test2", seed=42)  # Mismo seed
    config3 = BenchmarkConfig(name="test1", seed=43)  # Seed diferente
    
    hash1 = config1.hash()
    hash2 = config2.hash()
    hash3 = config3.hash()
    
    assert isinstance(hash1, str)
    assert len(hash1) == 12
    assert hash1 == hash2  # Parámetros relevantes iguales
    assert hash1 != hash3  # Seed diferente


def test_config_serialization():
    """Test de serialización JSON."""
    config = BenchmarkConfig(
        name="test",
        seed=42,
        n_runs=3,
    )
    
    # To dict
    config_dict = config.to_dict()
    assert config_dict["name"] == "test"
    assert "config_hash" in config_dict
    
    # From dict
    config_restored = BenchmarkConfig.from_dict(config_dict)
    assert config_restored.name == config.name
    assert config_restored.seed == config.seed


def test_config_validation():
    """Test de validación de parámetros."""
    # n_runs < 1
    with pytest.raises(ValueError):
        BenchmarkConfig(name="bad", n_runs=0)
    
    # train_test_split inválido
    with pytest.raises(ValueError):
        BenchmarkConfig(name="bad", train_test_split=1.5)


def test_preset_configs():
    """Test de configs pre-definidas."""
    assert "baseline_random" in BENCHMARK_CONFIGS
    assert "curriculum_softmax" in BENCHMARK_CONFIGS
    
    config = get_config("curriculum_softmax")
    assert config.use_curriculum is True


def test_list_configs():
    """Test de listar configs."""
    configs = list_configs()
    assert isinstance(configs, list)
    assert len(configs) > 0
    assert "curriculum_softmax" in configs


def test_create_custom_config():
    """Test de crear config personalizada."""
    config = create_custom_config(
        name="my_config",
        seed=123,
        n_runs=10,
    )
    
    assert config.name == "my_config"
    assert config.seed == 123
    assert config.n_runs == 10


# ============================================================================
# Tests de BenchmarkMetrics
# ============================================================================

def test_benchmark_metrics_creation():
    """Test de creación de métricas."""
    metrics = BenchmarkMetrics(
        final_loss=0.05,
        final_accuracy=0.9,
        best_loss=0.03,
        best_accuracy=0.95,
        total_epochs=50,
        total_training_time=120.5,
    )
    
    assert metrics.final_loss == 0.05
    assert metrics.total_epochs == 50


def test_metrics_serialization():
    """Test de serialización de métricas."""
    metrics = BenchmarkMetrics(
        final_loss=0.05,
        final_accuracy=0.9,
        best_loss=0.03,
        best_accuracy=0.95,
    )
    
    metrics_dict = metrics.to_dict()
    
    assert metrics_dict["final_loss"] == 0.05
    assert metrics_dict["final_accuracy"] == 0.9


def test_aggregated_metrics():
    """Test de agregación de métricas."""
    metrics_list = [
        BenchmarkMetrics(final_loss=0.05, final_accuracy=0.9, best_loss=0.03, best_accuracy=0.9),
        BenchmarkMetrics(final_loss=0.06, final_accuracy=0.85, best_loss=0.04, best_accuracy=0.85),
        BenchmarkMetrics(final_loss=0.04, final_accuracy=0.95, best_loss=0.02, best_accuracy=0.95),
    ]
    
    aggregated = BenchmarkMetrics.aggregate(metrics_list)
    
    assert isinstance(aggregated, AggregatedMetrics)
    assert aggregated.n_runs == 3
    
    # Verificar mean
    mean_loss = aggregated.get_mean("final_loss")
    assert abs(mean_loss - 0.05) < 0.01
    
    # Verificar std
    std_loss = aggregated.get_std("final_loss")
    assert std_loss > 0
    
    # Verificar CI
    ci_low, ci_high = aggregated.get_ci("final_loss")
    assert ci_low < mean_loss < ci_high


def test_format_metric():
    """Test de formateo de métricas."""
    metrics_list = [
        BenchmarkMetrics(final_loss=0.05, final_accuracy=0.9, best_loss=0.03, best_accuracy=0.9),
        BenchmarkMetrics(final_loss=0.06, final_accuracy=0.85, best_loss=0.04, best_accuracy=0.85),
    ]
    
    aggregated = BenchmarkMetrics.aggregate(metrics_list)
    formatted = aggregated.format_metric("final_loss", precision=4)
    
    assert isinstance(formatted, str)
    assert "±" in formatted
    assert "[" in formatted  # CI


# ============================================================================
# Tests de Helper Functions
# ============================================================================

def test_calculate_loss_stability():
    """Test de cálculo de estabilidad."""
    loss_history = [0.5, 0.4, 0.35, 0.3, 0.28]
    
    std, var, stability = calculate_loss_stability(loss_history)
    
    assert std > 0
    assert var > 0
    assert 0 <= stability <= 1


def test_calculate_loss_trend():
    """Test de cálculo de tendencia."""
    # Tendencia descendente
    loss_history = [0.5, 0.4, 0.3, 0.2, 0.1]
    slope = calculate_loss_trend(loss_history)
    
    assert slope < 0  # Descendente
    
    # Tendencia ascendente
    loss_history_up = [0.1, 0.2, 0.3, 0.4, 0.5]
    slope_up = calculate_loss_trend(loss_history_up)
    
    assert slope_up > 0  # Ascendente


def test_calculate_gate_consistency():
    """Test de consistencia de gates."""
    gates_history = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.5, 0.3, 0.2]),  # Idénticos
        np.array([0.5, 0.3, 0.2]),
    ]
    
    consistency = calculate_gate_consistency(gates_history)
    
    assert 0 <= consistency <= 1
    assert consistency > 0.9  # Muy consistente


def test_find_convergence_epoch():
    """Test de detección de convergencia."""
    loss_history = [0.5, 0.4, 0.3, 0.04, 0.03, 0.02, 0.02, 0.02]
    
    epoch = find_convergence_epoch(loss_history, threshold=0.05, patience=3)
    
    assert epoch is not None
    assert epoch >= 0


# ============================================================================
# Tests de Provenance
# ============================================================================

def test_provenance_capture():
    """Test de captura de provenance."""
    config = BenchmarkConfig(name="test", seed=42)
    
    provenance = BenchmarkProvenance.capture(config)
    
    assert provenance.run_id is not None
    assert provenance.config_hash == config.hash()
    assert provenance.python_version is not None
    assert provenance.numpy_version is not None


def test_provenance_serialization():
    """Test de serialización de provenance."""
    config = BenchmarkConfig(name="test", seed=42)
    provenance = BenchmarkProvenance.capture(config)
    
    # To dict
    prov_dict = provenance.to_dict()
    assert prov_dict["run_id"] == provenance.run_id
    
    # From dict
    prov_restored = BenchmarkProvenance.from_dict(prov_dict)
    assert prov_restored.run_id == provenance.run_id


def test_provenance_reproducibility_check():
    """Test de check de reproducibilidad."""
    config = BenchmarkConfig(name="test", seed=42)
    provenance = BenchmarkProvenance.capture(config)
    
    is_reproducible = provenance.is_reproducible()
    
    # Depende de si estamos en un repo git limpio
    assert isinstance(is_reproducible, bool)


# ============================================================================
# Tests de Baselines
# ============================================================================

def test_baseline_strategies():
    """Test de estrategias baseline."""
    assert "random_uniform" in BASELINE_STRATEGIES
    assert "equal" in BASELINE_STRATEGIES
    
    baseline = get_baseline("random_uniform")
    assert baseline.name == "random_uniform"


def test_list_baselines():
    """Test de listar baselines."""
    baselines = list_baselines()
    assert isinstance(baselines, list)
    assert len(baselines) > 0
    assert "random_uniform" in baselines


def test_baseline_reasoner():
    """Test de BaselineReasoner."""
    reasoner = BaselineReasoner(strategy="equal", n_blocks=3)
    
    # Generar gates
    state = np.random.rand(10)
    gates = reasoner.predict(state)
    
    assert gates.shape == (3,)
    assert np.allclose(gates.sum(), 1.0)  # Normalizados


def test_baseline_gates_generation():
    """Test de generación de gates."""
    baseline = get_baseline("equal")
    
    gates = baseline.generate_gates(n_blocks=5)
    
    assert gates.shape == (5,)
    assert np.allclose(gates, 0.2)  # Todos iguales (1/5)


# ============================================================================
# Tests de Comparator
# ============================================================================

def test_comparator_creation():
    """Test de creación del comparator."""
    comparator = BenchmarkComparator(confidence_level=0.95)
    
    assert comparator.confidence_level == 0.95
    assert comparator.alpha == 0.05


def test_compare_two():
    """Test de comparación pareada."""
    # Config A: mejor performance
    metrics_a = [
        BenchmarkMetrics(final_loss=0.02, final_accuracy=0.95, best_loss=0.02, best_accuracy=0.95),
        BenchmarkMetrics(final_loss=0.03, final_accuracy=0.93, best_loss=0.03, best_accuracy=0.93),
        BenchmarkMetrics(final_loss=0.025, final_accuracy=0.94, best_loss=0.025, best_accuracy=0.94),
    ]
    
    # Config B: peor performance
    metrics_b = [
        BenchmarkMetrics(final_loss=0.10, final_accuracy=0.80, best_loss=0.10, best_accuracy=0.80),
        BenchmarkMetrics(final_loss=0.12, final_accuracy=0.78, best_loss=0.12, best_accuracy=0.78),
        BenchmarkMetrics(final_loss=0.11, final_accuracy=0.79, best_loss=0.11, best_accuracy=0.79),
    ]
    
    comparator = BenchmarkComparator()
    
    comparison = comparator.compare_two(
        metrics_a,
        metrics_b,
        metric="final_loss",
        config_name_a="Config A",
        config_name_b="Config B",
    )
    
    assert isinstance(comparison, ComparisonResult)
    assert comparison.winner == "A"  # A es mejor (menor loss)
    assert comparison.mean_a < comparison.mean_b
    assert comparison.improvement > 0


def test_rank_configs():
    """Test de ranking."""
    results = {
        "config_a": [
            BenchmarkMetrics(final_loss=0.05, final_accuracy=0.9, best_loss=0.05, best_accuracy=0.9),
            BenchmarkMetrics(final_loss=0.06, final_accuracy=0.85, best_loss=0.06, best_accuracy=0.85),
        ],
        "config_b": [
            BenchmarkMetrics(final_loss=0.02, final_accuracy=0.95, best_loss=0.02, best_accuracy=0.95),
            BenchmarkMetrics(final_loss=0.03, final_accuracy=0.93, best_loss=0.03, best_accuracy=0.93),
        ],
    }
    
    comparator = BenchmarkComparator()
    ranking = comparator.rank_configs(results, metric="final_loss")
    
    assert len(ranking) == 2
    assert ranking[0][0] == "config_b"  # Mejor (menor loss)
    assert ranking[0][3] == 1  # Rank 1


# ============================================================================
# Tests de Integración
# ============================================================================

def test_full_config_workflow():
    """Test de workflow completo de config."""
    # Crear config
    config = create_custom_config(
        name="workflow_test",
        seed=42,
        n_runs=2,
    )
    
    # Serializar
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        config.to_json(Path(f.name))
        temp_path = Path(f.name)
    
    try:
        # Cargar
        config_loaded = BenchmarkConfig.from_json(temp_path)
        
        assert config_loaded.name == config.name
        assert config_loaded.seed == config.seed
        assert config_loaded.hash() == config.hash()
    
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
