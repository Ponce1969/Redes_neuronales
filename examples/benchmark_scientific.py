"""
Demo Científico Avanzado de Benchmark Suite.

Demuestra reproducibilidad, análisis estadístico y generación de reportes.

Uso:
    PYTHONPATH=src python examples/benchmark_scientific.py
"""

import sys
from pathlib import Path
import numpy as np

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.benchmark import (
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkComparator,
    verify_reproducibility,
    create_custom_config,
    ReportGenerator,
)
from core.reasoning.reasoner_manager import ReasonerManager
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock


def create_test_graph():
    """Crea grafo para tests."""
    graph = CognitiveGraphHybrid()
    
    sensor = CognitiveBlock(input_dim=8, hidden_dim=16, name="sensor")
    planner = CognitiveBlock(input_dim=16, hidden_dim=16, name="planner")
    decision = CognitiveBlock(input_dim=16, hidden_dim=8, name="decision")
    
    graph.add_block("sensor", sensor)
    graph.add_block("planner", planner)
    graph.add_block("decision", decision)
    
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")
    
    return graph


def main():
    """Ejecuta demo científico completo."""
    print("\n" + "="*70)
    print("🔬 COGNITIVE BENCHMARK SUITE - DEMO CIENTÍFICO")
    print("="*70)
    print("\nEste demo demuestra:")
    print("  ✅ Reproducibilidad completa (seeds, provenance)")
    print("  ✅ Análisis estadístico (t-tests, CI, effect size)")
    print("  ✅ Multi-run aggregation")
    print("  ✅ Reportes multi-formato")
    print()
    
    # Setup
    graph = create_test_graph()
    reasoner_manager = ReasonerManager(n_inputs=24, n_hidden=48, n_blocks=3)
    suite = BenchmarkSuite()
    
    # ========================================================================
    # Experimento 1: Reproducibilidad
    # ========================================================================
    
    print("="*70)
    print("🧪 EXPERIMENTO 1: REPRODUCIBILIDAD")
    print("="*70)
    print()
    
    # Crear config con seed específico
    config_repro = create_custom_config(
        name="reproducibility_test",
        description="Test de reproducibilidad con seed fijo",
        seed=42,
        use_curriculum=True,
        curriculum_type="fast",
        max_epochs_per_stage=20,
        n_runs=3,
    )
    
    print(f"Config creada: {config_repro.name}")
    print(f"Seed: {config_repro.seed}")
    print(f"Hash: {config_repro.hash()}")
    print()
    
    # Ejecutar benchmark
    print("Ejecutando benchmark...")
    result1 = suite.run_single(config_repro, reasoner_manager, graph, save_results=False)
    
    # Verificar provenance
    print("\n📋 Provenance:")
    print(result1.provenance.summary())
    print()
    
    # Verificar reproducibilidad
    repro_check = verify_reproducibility(result1.provenance)
    print("🔍 Verificación de reproducibilidad:")
    print(f"   Can reproduce: {repro_check['can_reproduce']}")
    if repro_check['warnings']:
        for warning in repro_check['warnings']:
            print(f"   ⚠️  {warning}")
    else:
        print("   ✅ Sin warnings")
    print()
    
    # ========================================================================
    # Experimento 2: Comparación Estadística
    # ========================================================================
    
    print("="*70)
    print("🔬 EXPERIMENTO 2: COMPARACIÓN ESTADÍSTICA")
    print("="*70)
    print()
    
    print("Configurando 3 estrategias:")
    print("  A. Curriculum + Softmax")
    print("  B. Curriculum + Top-K")
    print("  C. Baseline Random")
    print()
    
    configs = [
        create_custom_config(
            name="curriculum_softmax",
            use_curriculum=True,
            reasoner_mode="softmax",
            n_runs=5,
            max_epochs_per_stage=20,
            seed=42,
        ),
        create_custom_config(
            name="curriculum_topk",
            use_curriculum=True,
            reasoner_mode="topk",
            topk_value=2,
            n_runs=5,
            max_epochs_per_stage=20,
            seed=42,
        ),
        create_custom_config(
            name="baseline_random",
            use_curriculum=False,
            max_total_epochs=100,
            n_runs=5,
            seed=42,
        ),
    ]
    
    # Ejecutar comparación
    print("🚀 Ejecutando comparación (puede tardar algunos minutos)...")
    print()
    
    comparison_report = suite.run_comparison(
        configs=configs,
        reasoner_manager=reasoner_manager,
        graph=graph,
        metric="final_loss",
    )
    
    print()
    print("="*70)
    print("📊 RESULTADOS DE LA COMPARACIÓN")
    print("="*70)
    print()
    
    # Ranking
    print("🏆 RANKING:")
    for name, mean, std, rank in comparison_report.ranking:
        result = comparison_report.results[name]
        ci_low, ci_high = result.metrics.get_ci("final_loss")
        print(f"  {rank}. {name:25s} | {mean:.4f} ± {std:.4f} | CI [{ci_low:.4f}, {ci_high:.4f}]")
    
    print()
    
    # Análisis estadístico detallado
    print("🔬 ANÁLISIS ESTADÍSTICO:")
    print()
    
    comparator = BenchmarkComparator(confidence_level=0.95)
    
    for comp in comparison_report.comparisons:
        print(f"  {comp.config_a} vs {comp.config_b}:")
        print(f"    T-statistic: {comp.t_statistic:.3f}")
        print(f"    P-value: {comp.p_value:.4f} ({'✅ significant' if comp.significant else '⚠️  not significant'})")
        print(f"    Cohen's d: {comp.cohens_d:.3f} ({comp.effect_size_interpretation})")
        print(f"    Winner: {comp.winner} 🏆 (improvement: {comp.improvement:.1%})")
        print()
    
    # Friedman test (si hay suficientes configs)
    metrics_dict = {
        name: result.all_runs
        for name, result in comparison_report.results.items()
    }
    
    friedman_result = comparator.friedman_test(metrics_dict, metric="final_loss")
    
    print("📊 Friedman Test (múltiples grupos):")
    print(f"   Statistic: {friedman_result['statistic']:.3f}")
    print(f"   P-value: {friedman_result['p_value']:.4f}")
    print(f"   Significant: {'✅ Yes' if friedman_result['significant'] else '⚠️  No'}")
    print(f"   Interpretation: {friedman_result['interpretation']}")
    print()
    
    # ========================================================================
    # Experimento 3: Reportes Multi-Formato
    # ========================================================================
    
    print("="*70)
    print("📄 EXPERIMENTO 3: GENERACIÓN DE REPORTES")
    print("="*70)
    print()
    
    output_dir = Path("data/benchmarks/reports") / f"scientific_demo_{comparison_report.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Generando reportes en: {output_dir}")
    print()
    
    generator = ReportGenerator()
    generator.generate_all(
        comparison_report,
        output_dir,
        formats=["markdown", "html", "latex", "csv", "json"],
    )
    
    print("✅ Reportes generados:")
    print(f"   📝 Markdown: {output_dir}/report.md")
    print(f"   🌐 HTML: {output_dir}/report.html")
    print(f"   📊 LaTeX: {output_dir}/report.tex")
    print(f"   📈 CSV: {output_dir}/data.csv")
    print(f"   💾 JSON: {output_dir}/data.json")
    print()
    
    # Mostrar preview del Markdown
    print("📋 Preview del reporte Markdown:")
    print("-" * 70)
    
    md_path = output_dir / "report.md"
    if md_path.exists():
        md_content = md_path.read_text()
        # Mostrar primeras 30 líneas
        lines = md_content.split("\n")[:30]
        print("\n".join(lines))
        if len(md_content.split("\n")) > 30:
            print("...")
            print(f"(+{len(md_content.split('\n')) - 30} líneas más)")
    
    print("-" * 70)
    print()
    
    # ========================================================================
    # Resumen Final
    # ========================================================================
    
    print("="*70)
    print("🎉 DEMO CIENTÍFICO COMPLETADO")
    print("="*70)
    print()
    print("📊 Resumen de Experimentos:")
    print()
    print("  1️⃣ Reproducibilidad:")
    print(f"      ✅ Provenance completo capturado")
    print(f"      ✅ Git commit: {result1.provenance.git_commit[:8] if result1.provenance.git_commit else 'N/A'}")
    print(f"      ✅ Reproducible: {result1.provenance.is_reproducible()}")
    print()
    print("  2️⃣ Comparación Estadística:")
    print(f"      ✅ {len(configs)} configuraciones evaluadas")
    print(f"      ✅ {configs[0].n_runs} runs por configuración")
    print(f"      ✅ Confidence level: 95%")
    print(f"      ✅ Winner: {comparison_report.ranking[0][0]} 🏆")
    print()
    print("  3️⃣ Reportes:")
    print(f"      ✅ 5 formatos generados")
    print(f"      ✅ Listos para publicación científica")
    print()
    print("📚 Próximos Pasos:")
    print("  1. Revisar reportes: cat", str(output_dir / "report.md"))
    print("  2. Abrir HTML:", str(output_dir / "report.html"))
    print("  3. Analizar datos: pandas.read_csv('{}')".format(output_dir / "data.csv"))
    print("  4. Dashboard: PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py")
    print()


if __name__ == "__main__":
    main()
