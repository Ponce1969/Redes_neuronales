"""
Cognitive Benchmark Suite - Sistema profesional de benchmarking científico.

Este módulo implementa un sistema completo de benchmarking para comparar
configuraciones del Reasoner con rigor estadístico.

Componentes principales:
- configurations: BenchmarkConfig con reproducibilidad total
- metrics: Métricas científicas avanzadas
- provenance: Tracking completo de reproducibilidad
- baseline: Estrategias baseline para comparación
- comparator: Análisis estadístico (t-tests, CI, effect size)
- benchmark_suite: Runner principal
- report_generator: Reportes multi-formato

Uso básico:
    from core.benchmark import BenchmarkSuite, BENCHMARK_CONFIGS
    
    suite = BenchmarkSuite()
    result = suite.run_single(
        BENCHMARK_CONFIGS["curriculum_softmax"],
        reasoner_manager,
        graph
    )

Fase: 34
Autor: Neural Core Team
"""

from core.benchmark.configurations import (
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    get_config,
    list_configs,
    create_custom_config,
    create_baseline_random,
    create_curriculum_softmax,
    create_curriculum_topk,
    create_no_curriculum_topk,
    create_curriculum_fast,
    create_high_mutation,
    create_large_reasoner,
)

from core.benchmark.metrics import (
    BenchmarkMetrics,
    AggregatedMetrics,
    calculate_loss_stability,
    calculate_loss_trend,
    calculate_gate_consistency,
    find_convergence_epoch,
    calculate_dominant_gates,
)

from core.benchmark.provenance import (
    BenchmarkProvenance,
    verify_reproducibility,
)

from core.benchmark.baseline import (
    BaselineStrategy,
    BaselineReasoner,
    BASELINE_STRATEGIES,
    get_baseline,
    list_baselines,
    evaluate_baseline,
)

from core.benchmark.comparator import (
    BenchmarkComparator,
    ComparisonResult,
)

from core.benchmark.benchmark_suite import (
    BenchmarkSuite,
    BenchmarkResult,
    ComparisonReport,
)

from core.benchmark.report_generator import (
    ReportGenerator,
)


__all__ = [
    # Configurations
    "BenchmarkConfig",
    "BENCHMARK_CONFIGS",
    "get_config",
    "list_configs",
    "create_custom_config",
    "create_baseline_random",
    "create_curriculum_softmax",
    "create_curriculum_topk",
    "create_no_curriculum_topk",
    "create_curriculum_fast",
    "create_high_mutation",
    "create_large_reasoner",
    
    # Metrics
    "BenchmarkMetrics",
    "AggregatedMetrics",
    "calculate_loss_stability",
    "calculate_loss_trend",
    "calculate_gate_consistency",
    "find_convergence_epoch",
    "calculate_dominant_gates",
    
    # Provenance
    "BenchmarkProvenance",
    "verify_reproducibility",
    
    # Baselines
    "BaselineStrategy",
    "BaselineReasoner",
    "BASELINE_STRATEGIES",
    "get_baseline",
    "list_baselines",
    "evaluate_baseline",
    
    # Comparator
    "BenchmarkComparator",
    "ComparisonResult",
    
    # Suite
    "BenchmarkSuite",
    "BenchmarkResult",
    "ComparisonReport",
    
    # Reports
    "ReportGenerator",
]
