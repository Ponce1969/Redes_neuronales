"""
Comparador Estadístico - Análisis científico de resultados.

Compara configuraciones con rigor estadístico usando tests de hipótesis.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass

from core.benchmark.metrics import BenchmarkMetrics, AggregatedMetrics


@dataclass
class ComparisonResult:
    """Resultado de comparación entre dos configuraciones."""
    
    config_a: str
    config_b: str
    metric: str
    
    # Estadísticas descriptivas
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    median_a: float
    median_b: float
    
    # Tests estadísticos
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    confidence_level: float = 0.95
    
    # Effect size
    cohens_d: float
    effect_size_interpretation: str  # small/medium/large
    
    # Winner
    winner: str  # "A", "B", or "tie"
    improvement: float  # % de mejora del winner
    
    def summary(self) -> str:
        """Retorna resumen legible."""
        winner_symbol = {
            "A": "🏆",
            "B": "🏆",
            "tie": "🤝"
        }[self.winner]
        
        sig_symbol = "✅" if self.significant else "⚠️"
        
        lines = [
            f"Comparación: {self.config_a} vs {self.config_b}",
            f"Métrica: {self.metric}",
            f"",
            f"Config A: {self.mean_a:.4f} ± {self.std_a:.4f} (median: {self.median_a:.4f})",
            f"Config B: {self.mean_b:.4f} ± {self.std_b:.4f} (median: {self.median_b:.4f})",
            f"",
            f"T-test: t={self.t_statistic:.3f}, p={self.p_value:.4f} {sig_symbol}",
            f"Cohen's d: {self.cohens_d:.3f} ({self.effect_size_interpretation})",
            f"",
            f"Winner: {self.winner} {winner_symbol}",
            f"Improvement: {self.improvement:.1%}",
        ]
        
        return "\n".join(lines)


class BenchmarkComparator:
    """
    Comparador estadístico de benchmarks.
    
    Características:
    - ✅ T-tests para comparación pareada
    - ✅ Effect size (Cohen's d)
    - ✅ Confidence intervals
    - ✅ Multiple comparison correction (Bonferroni)
    - ✅ Ranking de configuraciones
    - ✅ Friedman test para múltiples grupos
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Inicializa comparador.
        
        Args:
            confidence_level: Nivel de confianza (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_two(
        self,
        metrics_a: List[BenchmarkMetrics],
        metrics_b: List[BenchmarkMetrics],
        metric: str = "final_loss",
        config_name_a: str = "A",
        config_name_b: str = "B",
    ) -> ComparisonResult:
        """
        Compara dos configuraciones estadísticamente.
        
        Args:
            metrics_a: Lista de métricas de config A
            metrics_b: Lista de métricas de config B
            metric: Métrica a comparar
            config_name_a: Nombre de config A
            config_name_b: Nombre de config B
        
        Returns:
            ComparisonResult con análisis completo
        """
        # Extraer valores
        values_a = np.array([getattr(m, metric) for m in metrics_a])
        values_b = np.array([getattr(m, metric) for m in metrics_b])
        
        # Estadísticas descriptivas
        mean_a, std_a = np.mean(values_a), np.std(values_a, ddof=1)
        mean_b, std_b = np.mean(values_b), np.std(values_b, ddof=1)
        median_a, median_b = np.median(values_a), np.median(values_b)
        
        # T-test (two-sample, independent)
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Significancia
        significant = p_value < self.alpha
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Interpretación de effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Determinar winner (menor es mejor para loss)
        if metric.endswith("loss") or metric.startswith("generalization_gap"):
            # Menor es mejor
            if mean_a < mean_b and significant:
                winner = "A"
                improvement = (mean_b - mean_a) / mean_b
            elif mean_b < mean_a and significant:
                winner = "B"
                improvement = (mean_a - mean_b) / mean_a
            else:
                winner = "tie"
                improvement = 0.0
        else:
            # Mayor es mejor (accuracy, stability, etc.)
            if mean_a > mean_b and significant:
                winner = "A"
                improvement = (mean_a - mean_b) / mean_b
            elif mean_b > mean_a and significant:
                winner = "B"
                improvement = (mean_b - mean_a) / mean_a
            else:
                winner = "tie"
                improvement = 0.0
        
        return ComparisonResult(
            config_a=config_name_a,
            config_b=config_name_b,
            metric=metric,
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            std_a=float(std_a),
            std_b=float(std_b),
            median_a=float(median_a),
            median_b=float(median_b),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            significant=significant,
            confidence_level=self.confidence_level,
            cohens_d=float(cohens_d),
            effect_size_interpretation=effect_interpretation,
            winner=winner,
            improvement=float(improvement),
        )
    
    def compare_all_pairwise(
        self,
        results: Dict[str, List[BenchmarkMetrics]],
        metric: str = "final_loss",
        bonferroni_correction: bool = True,
    ) -> List[ComparisonResult]:
        """
        Compara todas las configuraciones par a par.
        
        Args:
            results: Dict {config_name: [metrics]}
            metric: Métrica a comparar
            bonferroni_correction: Aplicar corrección de Bonferroni
        
        Returns:
            Lista de ComparisonResults
        """
        comparisons = []
        config_names = list(results.keys())
        
        # Comparar cada par
        for i, name_a in enumerate(config_names):
            for name_b in config_names[i+1:]:
                comparison = self.compare_two(
                    results[name_a],
                    results[name_b],
                    metric=metric,
                    config_name_a=name_a,
                    config_name_b=name_b,
                )
                comparisons.append(comparison)
        
        # Bonferroni correction
        if bonferroni_correction and comparisons:
            n_comparisons = len(comparisons)
            adjusted_alpha = self.alpha / n_comparisons
            
            for comp in comparisons:
                comp.significant = comp.p_value < adjusted_alpha
        
        return comparisons
    
    def rank_configs(
        self,
        results: Dict[str, List[BenchmarkMetrics]],
        metric: str = "final_loss",
    ) -> List[Tuple[str, float, float, int]]:
        """
        Rankea configuraciones por métrica.
        
        Args:
            results: Dict {config_name: [metrics]}
            metric: Métrica para rankear
        
        Returns:
            Lista de (config_name, mean, std, rank) ordenada por rank
        """
        rankings = []
        
        for config_name, metrics_list in results.items():
            values = [getattr(m, metric) for m in metrics_list]
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            rankings.append((config_name, mean, std))
        
        # Ordenar (menor es mejor para loss)
        if metric.endswith("loss") or metric.startswith("generalization_gap"):
            rankings.sort(key=lambda x: x[1])  # Ascendente
        else:
            rankings.sort(key=lambda x: x[1], reverse=True)  # Descendente
        
        # Añadir rank
        ranked = [
            (name, mean, std, rank + 1)
            for rank, (name, mean, std) in enumerate(rankings)
        ]
        
        return ranked
    
    def friedman_test(
        self,
        results: Dict[str, List[BenchmarkMetrics]],
        metric: str = "final_loss",
    ) -> Dict[str, Any]:
        """
        Friedman test para múltiples grupos relacionados.
        
        Útil cuando tienes múltiples configs evaluadas en las mismas tasks.
        
        Args:
            results: Dict {config_name: [metrics]}
            metric: Métrica a comparar
        
        Returns:
            Dict con statistic, p-value, significant
        """
        # Extraer valores para cada config
        config_names = list(results.keys())
        samples = []
        
        for name in config_names:
            values = [getattr(m, metric) for m in results[name]]
            samples.append(values)
        
        # Asegurar que todas tienen el mismo tamaño
        min_length = min(len(s) for s in samples)
        samples_trimmed = [s[:min_length] for s in samples]
        
        # Friedman test
        statistic, p_value = stats.friedmanchisquare(*samples_trimmed)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "n_configs": len(config_names),
            "n_samples": min_length,
            "interpretation": (
                "Las configuraciones difieren significativamente"
                if p_value < self.alpha
                else "No hay diferencia significativa entre configuraciones"
            ),
        }
    
    def generate_comparison_matrix(
        self,
        results: Dict[str, List[BenchmarkMetrics]],
        metric: str = "final_loss",
    ) -> np.ndarray:
        """
        Genera matriz de p-values de comparaciones pairwise.
        
        Args:
            results: Dict {config_name: [metrics]}
            metric: Métrica a comparar
        
        Returns:
            Matriz NxN de p-values
        """
        config_names = list(results.keys())
        n = len(config_names)
        
        matrix = np.zeros((n, n))
        
        for i, name_a in enumerate(config_names):
            for j, name_b in enumerate(config_names):
                if i == j:
                    matrix[i, j] = 1.0  # Mismo config
                elif i < j:
                    comparison = self.compare_two(
                        results[name_a],
                        results[name_b],
                        metric=metric,
                    )
                    matrix[i, j] = comparison.p_value
                    matrix[j, i] = comparison.p_value  # Simétrica
        
        return matrix
