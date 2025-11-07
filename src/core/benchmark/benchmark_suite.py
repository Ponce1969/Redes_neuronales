"""
Benchmark Suite Principal - Orquestador de experimentos científicos.

Runner principal que ejecuta benchmarks con reproducibilidad completa.
"""

import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from core.benchmark.configurations import BenchmarkConfig
from core.benchmark.metrics import (
    BenchmarkMetrics,
    AggregatedMetrics,
    calculate_loss_stability,
    calculate_loss_trend,
    calculate_gate_consistency,
    find_convergence_epoch,
    calculate_dominant_gates,
)
from core.benchmark.provenance import BenchmarkProvenance
from core.benchmark.comparator import BenchmarkComparator
from core.curriculum import CurriculumManager, create_standard_curriculum, tasks
from core.curriculum.metrics import CognitiveMetrics


@dataclass
class BenchmarkResult:
    """
    Resultado completo de un benchmark.
    
    Incluye config, métricas agregadas, provenance y runs individuales.
    """
    config: BenchmarkConfig
    metrics: AggregatedMetrics
    provenance: BenchmarkProvenance
    all_runs: List[BenchmarkMetrics] = field(default_factory=list)
    
    def save(self, path: Path):
        """Guarda resultado a archivo JSON."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "provenance": self.provenance.to_dict(),
            "all_runs": [m.to_dict() for m in self.all_runs],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'BenchmarkResult':
        """Carga resultado desde archivo JSON."""
        import json
        
        with open(path, "r") as f:
            data = json.load(f)
        
        from core.benchmark.configurations import BenchmarkConfig
        
        return cls(
            config=BenchmarkConfig.from_dict(data["config"]),
            metrics=AggregatedMetrics(**data["metrics"]),
            provenance=BenchmarkProvenance.from_dict(data["provenance"]),
            all_runs=[BenchmarkMetrics.from_dict(m) for m in data["all_runs"]],
        )


@dataclass
class ComparisonReport:
    """Reporte de comparación de múltiples benchmarks."""
    results: Dict[str, BenchmarkResult]
    comparisons: List[Any]  # ComparisonResults
    ranking: List[tuple]
    timestamp: datetime
    
    def save(self, output_dir: Path):
        """Guarda reporte completo."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar cada resultado
        for name, result in self.results.items():
            result.save(output_dir / f"{name}.json")
        
        # Guardar resumen
        summary = {
            "timestamp": self.timestamp.isoformat(),
            "n_configs": len(self.results),
            "config_names": list(self.results.keys()),
            "ranking": [
                {"rank": r[3], "name": r[0], "mean": r[1], "std": r[2]}
                for r in self.ranking
            ],
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


class BenchmarkSuite:
    """
    Suite principal de benchmarking.
    
    Características:
    - ✅ Reproducibilidad total (seeds, provenance)
    - ✅ Multi-run con agregación estadística
    - ✅ Integración con CurriculumManager
    - ✅ Métricas científicas avanzadas
    - ✅ Logging estructurado
    - ✅ Auto-save de resultados
    """
    
    def __init__(
        self,
        output_dir: str = "data/benchmarks/results",
        verbose: bool = True,
    ):
        """
        Inicializa benchmark suite.
        
        Args:
            output_dir: Directorio para guardar resultados
            verbose: Si True, imprime progreso
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
    
    def _log(self, message: str):
        """Log interno."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def run_single(
        self,
        config: BenchmarkConfig,
        reasoner_manager,
        graph,
        save_results: bool = True,
    ) -> BenchmarkResult:
        """
        Ejecuta un benchmark completo con N runs.
        
        Args:
            config: BenchmarkConfig a evaluar
            reasoner_manager: ReasonerManager instance
            graph: CognitiveGraphHybrid instance
            save_results: Si True, guarda resultados
        
        Returns:
            BenchmarkResult con métricas agregadas
        """
        self._log(f"\n{'='*70}")
        self._log(f"🧮 BENCHMARK: {config.name}")
        self._log(f"{'='*70}")
        
        # 1. Capturar provenance
        provenance = BenchmarkProvenance.capture(config)
        self._log(f"Run ID: {provenance.run_id}")
        self._log(f"Config hash: {config.hash()}")
        
        # 2. Set seed para reproducibilidad
        np.random.seed(config.seed)
        
        # 3. Ejecutar N runs
        self._log(f"\nEjecutando {config.n_runs} runs...")
        
        all_metrics = []
        
        for run_idx in range(config.n_runs):
            self._log(f"\n--- Run {run_idx + 1}/{config.n_runs} ---")
            
            # Seed único por run (pero reproducible)
            run_seed = config.seed + run_idx
            np.random.seed(run_seed)
            
            # Ejecutar single run
            metrics = self._run_once(
                config=config,
                reasoner_manager=reasoner_manager,
                graph=graph,
                run_idx=run_idx,
                provenance=provenance,
            )
            
            all_metrics.append(metrics)
            
            self._log(
                f"Run {run_idx + 1} completado: "
                f"loss={metrics.final_loss:.4f}, "
                f"acc={metrics.final_accuracy:.3f}"
            )
        
        # 4. Agregar resultados
        self._log("\nAgregando resultados...")
        aggregated = BenchmarkMetrics.aggregate(all_metrics)
        
        # 5. Crear resultado
        result = BenchmarkResult(
            config=config,
            metrics=aggregated,
            provenance=provenance,
            all_runs=all_metrics,
        )
        
        # 6. Guardar
        if save_results:
            output_path = self.output_dir / f"{provenance.run_id}.json"
            result.save(output_path)
            self._log(f"\n💾 Guardado en: {output_path}")
        
        # 7. Resumen
        self._log(f"\n{'='*70}")
        self._log(f"📊 RESUMEN: {config.name}")
        self._log(f"{'='*70}")
        self._log(f"Final loss: {aggregated.get_mean('final_loss'):.4f} ± {aggregated.get_std('final_loss'):.4f}")
        self._log(f"Best loss:  {aggregated.get_mean('best_loss'):.4f} ± {aggregated.get_std('best_loss'):.4f}")
        self._log(f"Accuracy:   {aggregated.get_mean('final_accuracy'):.3f} ± {aggregated.get_std('final_accuracy'):.3f}")
        self._log(f"Total epochs: {aggregated.get_mean('total_epochs'):.0f}")
        self._log(f"Training time: {aggregated.get_mean('total_training_time'):.1f}s")
        self._log(f"{'='*70}\n")
        
        return result
    
    def _run_once(
        self,
        config: BenchmarkConfig,
        reasoner_manager,
        graph,
        run_idx: int,
        provenance: BenchmarkProvenance,
    ) -> BenchmarkMetrics:
        """
        Ejecuta un run individual.
        
        Args:
            config: BenchmarkConfig
            reasoner_manager: ReasonerManager
            graph: CognitiveGraphHybrid
            run_idx: Índice del run
            provenance: Provenance del benchmark
        
        Returns:
            BenchmarkMetrics del run
        """
        start_time = time.time()
        
        # Reiniciar Reasoner
        from core.reasoning.reasoner import Reasoner
        reasoner_manager.reasoner = Reasoner(
            n_inputs=config.n_inputs,
            n_hidden=config.n_hidden,
            n_blocks=config.n_blocks,
            mode=config.reasoner_mode,
        )
        
        if config.use_curriculum:
            # Usar CurriculumManager
            curriculum_mgr = CurriculumManager(
                reasoner_manager=reasoner_manager,
                graph=graph,
                auto_save=False,  # No auto-save durante benchmark
            )
            
            # Añadir etapas según config
            if config.curriculum_type == "standard":
                stages = create_standard_curriculum()
            elif config.curriculum_type == "fast":
                # Curriculum más rápido
                stages = create_standard_curriculum()[:4]  # Solo primeras 4
            else:
                stages = create_standard_curriculum()
            
            # Ajustar epochs según config
            for stage in stages:
                stage.max_epochs = config.max_epochs_per_stage
                stage.success_threshold = config.success_threshold
                stage.fail_threshold = config.fail_threshold
                curriculum_mgr.add_stage(stage)
            
            # Ejecutar curriculum
            history = curriculum_mgr.run()
            
            # Extraer métricas
            total_epochs = sum(r['epochs'] for r in history)
            final_loss = history[-1]['mse_loss'] if history else np.inf
            final_accuracy = history[-1].get('accuracy', 0.0) if history else 0.0
            best_loss = min(r['mse_loss'] for r in history) if history else np.inf
            stages_completed = len(history)
            
            # Loss history (aproximado)
            loss_history = [r['mse_loss'] for r in history]
            
            # Gates history (si disponible)
            gates_history = []
            
        else:
            # Sin curriculum: entrenar en task único (XOR como default)
            X, Y = tasks.xor_task(samples=config.batch_size * 4)
            
            from core.curriculum import CurriculumEvaluator
            evaluator = CurriculumEvaluator(graph, mode=config.reasoner_mode)
            
            loss_history = []
            gates_history = []
            best_loss = np.inf
            
            for epoch in range(config.max_total_epochs):
                # Evaluar
                metrics_dict = evaluator.evaluate(
                    reasoner_manager.reasoner,
                    X,
                    Y,
                    track_gates=True,
                )
                
                loss = metrics_dict['mse_loss']
                loss_history.append(loss)
                
                if evaluator.gates_history:
                    gates_history = evaluator.gates_history
                
                if loss < best_loss:
                    best_loss = loss
                
                # Early stopping
                if loss < config.success_threshold:
                    break
                
                # Evolucionar
                child = reasoner_manager.reasoner.mutate(scale=config.mutation_scale)
                child_loss = evaluator.evaluate(child, X, Y, track_gates=False)['mse_loss']
                
                if child_loss < loss:
                    reasoner_manager.reasoner = child
            
            total_epochs = len(loss_history)
            final_loss = loss_history[-1] if loss_history else np.inf
            final_accuracy = metrics_dict.get('accuracy', 0.0)
            stages_completed = None
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calcular métricas avanzadas
        loss_std, loss_var, stability = calculate_loss_stability(loss_history)
        loss_slope = calculate_loss_trend(loss_history)
        convergence_epoch = find_convergence_epoch(loss_history, config.success_threshold)
        
        # Gates metrics
        if gates_history:
            gate_diversity = CognitiveMetrics.gate_diversity(gates_history)
            gate_entropy = CognitiveMetrics.gate_entropy(gates_history)
            gate_consistency = calculate_gate_consistency(gates_history)
            gate_utilization = CognitiveMetrics.gate_utilization(gates_history)
            dominant_gates = calculate_dominant_gates(gates_history, top_n=3)
        else:
            gate_diversity = gate_entropy = gate_consistency = gate_utilization = 0.0
            dominant_gates = []
        
        # Crear métricas
        metrics = BenchmarkMetrics(
            # Performance
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            best_loss=best_loss,
            best_accuracy=final_accuracy,  # Simplificado
            # Convergencia
            convergence_epoch=convergence_epoch,
            time_to_threshold=None,  # TODO
            converged=convergence_epoch is not None,
            # Estabilidad
            loss_std=loss_std,
            loss_variance=loss_var,
            training_stability=stability,
            loss_trend_slope=loss_slope,
            # Gates
            gate_diversity=gate_diversity,
            gate_entropy=gate_entropy,
            gate_consistency=gate_consistency,
            gate_utilization=gate_utilization,
            dominant_gates=dominant_gates,
            # Eficiencia
            total_epochs=total_epochs,
            total_training_time=training_time,
            epochs_per_second=total_epochs / training_time if training_time > 0 else 0.0,
            # Generalización (simplificado)
            train_loss=final_loss,
            test_loss=final_loss,  # TODO: split train/test
            generalization_gap=0.0,
            # Curriculum
            stages_completed=stages_completed,
            # Meta
            run_id=f"{provenance.run_id}_run{run_idx}",
            config_hash=config.hash(),
            timestamp=datetime.now().isoformat(),
        )
        
        return metrics
    
    def run_comparison(
        self,
        configs: List[BenchmarkConfig],
        reasoner_manager,
        graph,
        metric: str = "final_loss",
    ) -> ComparisonReport:
        """
        Ejecuta múltiples configs y genera reporte comparativo.
        
        Args:
            configs: Lista de BenchmarkConfig a comparar
            reasoner_manager: ReasonerManager instance
            graph: CognitiveGraphHybrid instance
            metric: Métrica principal para comparación
        
        Returns:
            ComparisonReport con análisis completo
        """
        self._log(f"\n🚀 Iniciando comparación de {len(configs)} configuraciones")
        self._log(f"Métrica principal: {metric}\n")
        
        # Ejecutar cada config
        results = {}
        
        for config in configs:
            result = self.run_single(config, reasoner_manager, graph)
            results[config.name] = result
        
        # Comparación estadística
        self._log("\n📊 Análisis estadístico...")
        
        comparator = BenchmarkComparator()
        
        # Extraer metrics lists
        metrics_dict = {
            name: result.all_runs
            for name, result in results.items()
        }
        
        # Comparaciones pairwise
        comparisons = comparator.compare_all_pairwise(metrics_dict, metric)
        
        # Ranking
        ranking = comparator.rank_configs(metrics_dict, metric)
        
        # Crear reporte
        report = ComparisonReport(
            results=results,
            comparisons=comparisons,
            ranking=ranking,
            timestamp=datetime.now(),
        )
        
        # Imprimir ranking
        self._log("\n🏆 RANKING FINAL:")
        for name, mean, std, rank in ranking:
            self._log(f"  {rank}. {name:25s} | {mean:.4f} ± {std:.4f}")
        
        return report
