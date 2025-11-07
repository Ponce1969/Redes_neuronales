"""
Demo Básico de Benchmark Suite.

Muestra cómo ejecutar un benchmark simple y comparar resultados.

Uso:
    PYTHONPATH=src python examples/benchmark_demo.py
"""

import sys
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.benchmark import (
    BenchmarkSuite,
    BENCHMARK_CONFIGS,
    list_configs,
    ReportGenerator,
)
from core.reasoning.reasoner_manager import ReasonerManager
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock


def create_simple_graph():
    """Crea un grafo cognitivo simple para benchmarking."""
    graph = CognitiveGraphHybrid()
    
    # Bloques básicos
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
    """Ejecuta el demo de benchmark."""
    print("="*70)
    print("🧮 COGNITIVE BENCHMARK SUITE - DEMO BÁSICO")
    print("="*70)
    print()
    
    # Mostrar configs disponibles
    print("📋 Configuraciones disponibles:")
    for config_name in list_configs():
        print(f"   - {config_name}")
    print()
    
    # 1. Crear grafo cognitivo
    print("📊 Creando grafo cognitivo...")
    graph = create_simple_graph()
    
    # 2. Crear ReasonerManager
    print("🧠 Inicializando Reasoner...")
    reasoner_manager = ReasonerManager(
        n_inputs=24,
        n_hidden=48,
        n_blocks=3,
    )
    
    # 3. Crear BenchmarkSuite
    print("🧮 Configurando benchmark suite...")
    suite = BenchmarkSuite(
        output_dir="data/benchmarks/results",
        verbose=True,
    )
    
    print()
    print("="*70)
    print("🚀 EJECUTANDO BENCHMARK ÚNICO")
    print("="*70)
    print()
    
    # 4. Ejecutar un benchmark (rápido para demo)
    config = BENCHMARK_CONFIGS["curriculum_fast"]
    
    result = suite.run_single(
        config=config,
        reasoner_manager=reasoner_manager,
        graph=graph,
        save_results=True,
    )
    
    print()
    print("="*70)
    print("✅ BENCHMARK COMPLETADO")
    print("="*70)
    print()
    print(f"Config: {config.name}")
    print(f"Runs: {result.metrics.n_runs}")
    print(f"Final loss: {result.metrics.format_metric('final_loss')}")
    print(f"Accuracy: {result.metrics.format_metric('final_accuracy', precision=3)}")
    print(f"Total epochs: {result.metrics.get_mean('total_epochs'):.0f}")
    print(f"Training time: {result.metrics.get_mean('total_training_time'):.1f}s")
    print()
    print(f"💾 Resultado guardado: data/benchmarks/results/{result.provenance.run_id}.json")
    print()
    
    # Opcional: ejecutar comparación
    print("="*70)
    print("⚖️ COMPARACIÓN DE MÚLTIPLES CONFIGS (Opcional)")
    print("="*70)
    print()
    
    run_comparison = input("¿Ejecutar comparación de 2 configs? (toma ~5 min) [y/N]: ").lower() == 'y'
    
    if run_comparison:
        print()
        print("🚀 Ejecutando comparación...")
        print("   1. curriculum_fast")
        print("   2. baseline_random")
        print()
        
        comparison_report = suite.run_comparison(
            configs=[
                BENCHMARK_CONFIGS["curriculum_fast"],
                BENCHMARK_CONFIGS["baseline_random"],
            ],
            reasoner_manager=reasoner_manager,
            graph=graph,
            metric="final_loss",
        )
        
        print()
        print("="*70)
        print("📊 RESULTADOS DE LA COMPARACIÓN")
        print("="*70)
        print()
        
        # Mostrar ranking
        print("🏆 Ranking:")
        for name, mean, std, rank in comparison_report.ranking:
            print(f"   {rank}. {name:20s} | {mean:.4f} ± {std:.4f}")
        
        print()
        
        # Mostrar comparación estadística
        if comparison_report.comparisons:
            comp = comparison_report.comparisons[0]
            print("🔬 Análisis Estadístico:")
            print(f"   Winner: {comp.winner} 🏆")
            print(f"   Improvement: {comp.improvement:.1%}")
            print(f"   Significant: {'✅ Yes' if comp.significant else '⚠️  No'} (p={comp.p_value:.4f})")
            print(f"   Effect size: {comp.cohens_d:.3f} ({comp.effect_size_interpretation})")
        
        print()
        
        # Generar reportes
        output_dir = Path("data/benchmarks/reports") / f"demo_{comparison_report.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📄 Generando reportes en: {output_dir}")
        
        generator = ReportGenerator()
        generator.generate_all(comparison_report, output_dir)
        
        print()
        print("✅ Reportes generados:")
        print(f"   - {output_dir}/report.md")
        print(f"   - {output_dir}/report.html")
        print(f"   - {output_dir}/data.csv")
        print(f"   - {output_dir}/data.json")
    
    print()
    print("="*70)
    print("🎉 DEMO COMPLETADO")
    print("="*70)
    print()
    print("Próximos pasos:")
    print("  1. Ver resultados: cat data/benchmarks/reports/demo_*/report.md")
    print("  2. Abrir dashboard: PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py")
    print("  3. Ejecutar más benchmarks vía API: curl -X POST http://localhost:8000/benchmark/run")
    print()


if __name__ == "__main__":
    main()
