"""
Métricas cognitivas avanzadas para evaluación de Curriculum Learning.

Incluye métricas de performance, convergencia y diversidad.
"""

import numpy as np
from typing import Dict, List, Optional


class CognitiveMetrics:
    """
    Calculador de métricas avanzadas para evaluar el progreso cognitivo.
    
    Métricas implementadas:
    - MSE Loss: Error cuadrático medio
    - MAE Loss: Error absoluto medio
    - Accuracy: Precisión en clasificación binaria
    - Gate Diversity: Cuán uniformemente usa los bloques
    - Gate Entropy: Entropía de Shannon de los gates
    - Convergence Rate: Velocidad de mejora
    - Stability: Estabilidad del error (1 / varianza)
    """
    
    @staticmethod
    def compute_all(
        predictions: np.ndarray,
        targets: np.ndarray,
        gates_history: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Calcula todas las métricas disponibles.
        
        Args:
            predictions: Predicciones del modelo (samples, outputs)
            targets: Valores verdaderos (samples, outputs)
            gates_history: Historial de gates aplicados (opcional)
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        metrics = {}
        
        # Asegurar shapes compatibles
        predictions = np.atleast_2d(predictions)
        targets = np.atleast_2d(targets)
        
        if predictions.shape != targets.shape:
            # Broadcast si es necesario
            if targets.shape[1] == 1 and predictions.shape[1] > 1:
                targets = np.repeat(targets, predictions.shape[1], axis=1)
        
        # 1. MSE Loss (principal)
        metrics['mse_loss'] = CognitiveMetrics.mse_loss(predictions, targets)
        
        # 2. MAE Loss (alternativa)
        metrics['mae_loss'] = CognitiveMetrics.mae_loss(predictions, targets)
        
        # 3. Accuracy (solo si es binario)
        if CognitiveMetrics._is_binary_task(targets):
            metrics['accuracy'] = CognitiveMetrics.accuracy(predictions, targets)
        
        # 4. Métricas de gates (si están disponibles)
        if gates_history:
            gate_metrics = CognitiveMetrics.gate_metrics(gates_history)
            metrics.update(gate_metrics)
        
        # 5. Convergence Rate (si hay suficientes samples)
        if len(predictions) >= 20:
            metrics['convergence_rate'] = CognitiveMetrics.convergence_rate(
                predictions, targets
            )
        
        # 6. Stability (varianza del error)
        metrics['stability'] = CognitiveMetrics.stability(predictions, targets)
        
        return metrics
    
    @staticmethod
    def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calcula Mean Squared Error."""
        return float(np.mean((predictions - targets) ** 2))
    
    @staticmethod
    def mae_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calcula Mean Absolute Error."""
        return float(np.mean(np.abs(predictions - targets)))
    
    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calcula accuracy para clasificación binaria.
        
        Args:
            predictions: Predicciones continuas
            targets: Targets binarios (0 o 1)
            threshold: Umbral para convertir predicciones a binario
        
        Returns:
            Accuracy como fracción [0, 1]
        """
        preds_binary = (predictions > threshold).astype(float)
        targets_binary = (targets > threshold).astype(float)
        return float(np.mean(preds_binary == targets_binary))
    
    @staticmethod
    def gate_metrics(gates_history: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcula métricas de diversidad y uso de gates.
        
        Args:
            gates_history: Lista de arrays de gates (cada uno shape: (n_blocks,))
        
        Returns:
            Diccionario con gate_diversity y gate_entropy
        """
        if not gates_history:
            return {}
        
        # Convertir a array 2D: (timesteps, n_blocks)
        gates_array = np.array(gates_history)
        
        # Media de gates por bloque a lo largo del tiempo
        gates_avg = np.mean(gates_array, axis=0)
        
        # Diversity: 1 - std (cuán uniforme es el uso)
        # Si todos los bloques se usan igual → diversity alta
        diversity = float(1.0 - np.std(gates_avg))
        
        # Entropy de Shannon: mide "sorpresa" o uniformidad
        # Alta entropía = distribución uniforme
        epsilon = 1e-10  # Para evitar log(0)
        probs = gates_avg / (gates_avg.sum() + epsilon)
        entropy = float(-np.sum(probs * np.log(probs + epsilon)))
        
        # Utilization: porcentaje de bloques realmente usados (>0.1)
        active_blocks = np.sum(gates_avg > 0.1)
        total_blocks = len(gates_avg)
        utilization = float(active_blocks / total_blocks)
        
        return {
            'gate_diversity': diversity,
            'gate_entropy': entropy,
            'gate_utilization': utilization,
        }
    
    @staticmethod
    def convergence_rate(predictions: np.ndarray, targets: np.ndarray, split: int = 10) -> float:
        """
        Calcula la tasa de convergencia comparando primeros vs últimos samples.
        
        Un valor positivo indica que el modelo mejoró durante el entrenamiento.
        
        Args:
            predictions: Predicciones ordenadas temporalmente
            targets: Targets correspondientes
            split: Cuántos samples usar para primera y última ventana
        
        Returns:
            Mejora en MSE (positivo = mejoró)
        """
        first_loss = np.mean((predictions[:split] - targets[:split]) ** 2)
        last_loss = np.mean((predictions[-split:] - targets[-split:]) ** 2)
        return float(first_loss - last_loss)  # Positivo = mejoró
    
    @staticmethod
    def stability(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcula estabilidad como 1 / (1 + varianza del error).
        
        Alta estabilidad = predicciones consistentes.
        
        Args:
            predictions: Predicciones del modelo
            targets: Targets verdaderos
        
        Returns:
            Estabilidad en [0, 1] (1 = muy estable)
        """
        errors = (predictions - targets) ** 2
        variance = np.var(errors)
        return float(1.0 / (1.0 + variance))
    
    @staticmethod
    def _is_binary_task(targets: np.ndarray) -> bool:
        """Detecta si la tarea es clasificación binaria."""
        unique_vals = np.unique(targets)
        return len(unique_vals) <= 2 and np.all((unique_vals == 0) | (unique_vals == 1))
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
        """
        Formatea métricas para display legible.
        
        Args:
            metrics: Diccionario de métricas
            precision: Decimales a mostrar
        
        Returns:
            String formateado
        """
        lines = []
        for key, value in metrics.items():
            if 'accuracy' in key or 'utilization' in key:
                # Mostrar como porcentaje
                lines.append(f"{key}: {value:.1%}")
            else:
                lines.append(f"{key}: {value:.{precision}f}")
        return " | ".join(lines)
