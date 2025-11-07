"""
Generador de Reportes - Multi-formato para análisis científico.

Genera reportes en Markdown, HTML, LaTeX, CSV y JSON.
"""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import json
import csv

from core.benchmark.benchmark_suite import BenchmarkResult, ComparisonReport


class ReportGenerator:
    """
    Generador de reportes multi-formato.
    
    Formatos soportados:
    - ✅ Markdown (legible, versionable)
    - ✅ HTML (interactivo con gráficos)
    - ✅ LaTeX (papers científicos)
    - ✅ CSV (análisis en Excel/Pandas)
    - ✅ JSON (programático)
    """
    
    def generate_all(
        self,
        report: ComparisonReport,
        output_dir: Path,
        formats: List[str] = None,
    ):
        """
        Genera todos los formatos.
        
        Args:
            report: ComparisonReport con resultados
            output_dir: Directorio de salida
            formats: Lista de formatos (None = todos)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ["markdown", "html", "csv", "json"]
        
        if "markdown" in formats:
            self.generate_markdown(report, output_dir / "report.md")
        
        if "html" in formats:
            self.generate_html(report, output_dir / "report.html")
        
        if "latex" in formats:
            self.generate_latex(report, output_dir / "report.tex")
        
        if "csv" in formats:
            self.generate_csv(report, output_dir / "data.csv")
        
        if "json" in formats:
            self.generate_json(report, output_dir / "data.json")
        
        print(f"✅ Reportes generados en: {output_dir}")
    
    def generate_markdown(self, report: ComparisonReport, path: Path):
        """Genera reporte en Markdown."""
        lines = []
        
        # Header
        lines.append("# 🧮 Benchmark Comparison Report")
        lines.append("")
        lines.append(f"**Generated**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Configurations**: {len(report.results)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Ranking
        lines.append("## 🏆 Ranking")
        lines.append("")
        lines.append("| Rank | Configuration | Mean | Std | CI (95%) |")
        lines.append("|------|---------------|------|-----|----------|")
        
        for name, mean, std, rank in report.ranking:
            # Obtener CI del resultado
            result = report.results[name]
            ci_low, ci_high = result.metrics.get_ci("final_loss")
            
            lines.append(
                f"| {rank} | {name} | {mean:.4f} | {std:.4f} | "
                f"[{ci_low:.4f}, {ci_high:.4f}] |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Resultados detallados
        lines.append("## 📊 Detailed Results")
        lines.append("")
        
        for name, result in report.results.items():
            lines.append(f"### {name}")
            lines.append("")
            
            # Config info
            lines.append(f"**Config Hash**: `{result.config.hash()}`")
            lines.append(f"**Runs**: {result.metrics.n_runs}")
            lines.append("")
            
            # Key metrics
            metrics = result.metrics
            lines.append("**Performance**:")
            lines.append(f"- Final Loss: {metrics.format_metric('final_loss')}")
            lines.append(f"- Best Loss: {metrics.format_metric('best_loss')}")
            lines.append(f"- Accuracy: {metrics.format_metric('final_accuracy', precision=3)}")
            lines.append("")
            
            lines.append("**Efficiency**:")
            lines.append(f"- Total Epochs: {metrics.get_mean('total_epochs'):.0f}")
            lines.append(f"- Training Time: {metrics.get_mean('total_training_time'):.1f}s")
            lines.append(f"- Epochs/sec: {metrics.get_mean('epochs_per_second'):.2f}")
            lines.append("")
            
            lines.append("**Stability**:")
            lines.append(f"- Training Stability: {metrics.format_metric('training_stability')}")
            lines.append(f"- Loss Std: {metrics.format_metric('loss_std')}")
            lines.append("")
            
            lines.append("**Gates**:")
            lines.append(f"- Diversity: {metrics.format_metric('gate_diversity')}")
            lines.append(f"- Entropy: {metrics.format_metric('gate_entropy')}")
            lines.append(f"- Consistency: {metrics.format_metric('gate_consistency')}")
            lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Comparaciones estadísticas
        if report.comparisons:
            lines.append("## 🔬 Statistical Comparisons")
            lines.append("")
            
            for comp in report.comparisons:
                lines.append(f"### {comp.config_a} vs {comp.config_b}")
                lines.append("")
                
                winner_symbol = "🏆" if comp.winner in ["A", "B"] else "🤝"
                sig_symbol = "✅" if comp.significant else "⚠️"
                
                lines.append(f"- **Winner**: {comp.winner} {winner_symbol}")
                lines.append(f"- **Improvement**: {comp.improvement:.1%}")
                lines.append(f"- **Significant**: {comp.significant} {sig_symbol} (p={comp.p_value:.4f})")
                lines.append(f"- **Effect Size**: {comp.cohens_d:.3f} ({comp.effect_size_interpretation})")
                lines.append("")
        
        # Escribir
        path.write_text("\n".join(lines))
    
    def generate_html(self, report: ComparisonReport, path: Path):
        """Genera reporte en HTML interactivo."""
        html = []
        
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <title>Benchmark Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("    h1 { color: #333; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        html.append("    th { background-color: #4CAF50; color: white; }")
        html.append("    tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("    .metric { font-family: monospace; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append("  <h1>🧮 Benchmark Comparison Report</h1>")
        html.append(f"  <p><strong>Generated</strong>: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"  <p><strong>Configurations</strong>: {len(report.results)}</p>")
        
        # Ranking table
        html.append("  <h2>🏆 Ranking</h2>")
        html.append("  <table>")
        html.append("    <tr><th>Rank</th><th>Configuration</th><th>Mean</th><th>Std</th></tr>")
        
        for name, mean, std, rank in report.ranking:
            html.append(f"    <tr>")
            html.append(f"      <td>{rank}</td>")
            html.append(f"      <td><strong>{name}</strong></td>")
            html.append(f"      <td class='metric'>{mean:.4f}</td>")
            html.append(f"      <td class='metric'>{std:.4f}</td>")
            html.append(f"    </tr>")
        
        html.append("  </table>")
        
        # Detailed results
        html.append("  <h2>📊 Detailed Results</h2>")
        
        for name, result in report.results.items():
            html.append(f"  <h3>{name}</h3>")
            html.append(f"  <p><strong>Config Hash</strong>: {result.config.hash()}</p>")
            html.append(f"  <p><strong>Runs</strong>: {result.metrics.n_runs}</p>")
            
            html.append("  <table>")
            html.append("    <tr><th>Metric</th><th>Value</th></tr>")
            
            metrics_to_show = [
                ("Final Loss", "final_loss"),
                ("Best Loss", "best_loss"),
                ("Accuracy", "final_accuracy"),
                ("Total Epochs", "total_epochs"),
                ("Training Time (s)", "total_training_time"),
                ("Gate Diversity", "gate_diversity"),
                ("Gate Entropy", "gate_entropy"),
            ]
            
            for label, metric in metrics_to_show:
                value = result.metrics.format_metric(metric, precision=4)
                html.append(f"    <tr><td>{label}</td><td class='metric'>{value}</td></tr>")
            
            html.append("  </table>")
        
        html.append("</body>")
        html.append("</html>")
        
        path.write_text("\n".join(html))
    
    def generate_latex(self, report: ComparisonReport, path: Path):
        """Genera reporte en LaTeX para papers."""
        lines = []
        
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{amsmath}")
        lines.append(r"\begin{document}")
        lines.append("")
        lines.append(r"\section{Benchmark Results}")
        lines.append("")
        
        # Ranking table
        lines.append(r"\subsection{Configuration Ranking}")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{clcc}")
        lines.append(r"\toprule")
        lines.append(r"Rank & Configuration & Mean & Std \\")
        lines.append(r"\midrule")
        
        for name, mean, std, rank in report.ranking:
            # Escapar underscores
            name_escaped = name.replace("_", r"\_")
            lines.append(f"{rank} & {name_escaped} & {mean:.4f} & {std:.4f} \\\\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Benchmark configuration ranking by final loss.}")
        lines.append(r"\end{table}")
        lines.append("")
        
        lines.append(r"\end{document}")
        
        path.write_text("\n".join(lines))
    
    def generate_csv(self, report: ComparisonReport, path: Path):
        """Genera CSV con todas las métricas."""
        rows = []
        
        # Header
        header = [
            "config_name", "rank", "n_runs",
            "final_loss_mean", "final_loss_std", "final_loss_ci_low", "final_loss_ci_high",
            "best_loss_mean", "best_loss_std",
            "accuracy_mean", "accuracy_std",
            "total_epochs_mean", "training_time_mean",
            "gate_diversity_mean", "gate_entropy_mean",
            "training_stability_mean",
        ]
        rows.append(header)
        
        # Data rows
        for name, mean, std, rank in report.ranking:
            result = report.results[name]
            metrics = result.metrics
            
            ci_low, ci_high = metrics.get_ci("final_loss")
            
            row = [
                name,
                rank,
                metrics.n_runs,
                metrics.get_mean("final_loss"),
                metrics.get_std("final_loss"),
                ci_low,
                ci_high,
                metrics.get_mean("best_loss"),
                metrics.get_std("best_loss"),
                metrics.get_mean("final_accuracy"),
                metrics.get_std("final_accuracy"),
                metrics.get_mean("total_epochs"),
                metrics.get_mean("total_training_time"),
                metrics.get_mean("gate_diversity"),
                metrics.get_mean("gate_entropy"),
                metrics.get_mean("training_stability"),
            ]
            rows.append(row)
        
        # Escribir
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def generate_json(self, report: ComparisonReport, path: Path):
        """Genera JSON completo."""
        data = {
            "timestamp": report.timestamp.isoformat(),
            "n_configs": len(report.results),
            "ranking": [
                {
                    "rank": r[3],
                    "name": r[0],
                    "mean": r[1],
                    "std": r[2],
                }
                for r in report.ranking
            ],
            "results": {
                name: {
                    "config_hash": result.config.hash(),
                    "n_runs": result.metrics.n_runs,
                    "metrics": result.metrics.to_dict(),
                }
                for name, result in report.results.items()
            },
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
