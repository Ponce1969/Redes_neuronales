"""
Trainer RL simple para auto-curriculum learning.
Implementa REINFORCE para challenger y entrenamiento supervisado para reasoner.
"""

from __future__ import annotations
from typing import List, Tuple, Callable, Dict, Any
import random
import math
from core.network import NeuralNetwork
from core.latent import LatentSpace, sample_gaussian_z
from core import losses


class SimpleRLTrainer:
    """
    Trainer que implementa auto-curriculum mediante:
    1. Challenger: genera tareas usando z latente
    2. Reasoner: resuelve tareas con ayuda de z
    3. Recompensas: basadas en dificultad y diversidad
    """
    
    def __init__(
        self,
        challenger: NeuralNetwork,
        reasoner: NeuralNetwork,
        latent_dim: int = 8,
        lr_challenger: float = 0.01,
        lr_reasoner: float = 0.01,
        baseline_decay: float = 0.9
    ):
        self.challenger = challenger
        self.reasoner = reasoner
        self.latent_space = LatentSpace(latent_dim)
        self.lr_challenger = lr_challenger
        self.lr_reasoner = lr_reasoner
        self.baseline_decay = baseline_decay
        
        # Tracking para estabilidad
        self.reward_baseline = 0.0
        self.episode_count = 0
        self.task_history: List[Dict[str, Any]] = []
    
    def generate_task(self, z: List[float] | None = None) -> Tuple[List[float], float]:
        """
        Genera una tarea usando el challenger
        Returns: (task_vector, log_probability)
        """
        if z is None:
            z = self.latent_space.sample()
        
        # Challenger genera la tarea
        task = self.challenger.forward(z)
        
        # Log-probability simplificado (para REINFORCE)
        # En una implementación real, esto vendría de una distribución paramétrica
        log_prob = -sum(x**2 for x in task) / (2 * len(task))  # Distribución gaussiana implícita
        
        return task, log_prob
    
    def solve_task(self, task: List[float], z: List[float] | None = None) -> List[float]:
        """Reasoner resuelve la tarea"""
        if z is None:
            z = self.latent_space.sample()
        
        # Concatenar tarea y z para el reasoner
        combined_input = task + z
        return self.reasoner.forward(combined_input)
    
    def evaluate_task(self, task: List[float], prediction: List[float], target: List[float]) -> Dict[str, float]:
        """
        Evalúa la tarea y calcula recompensas
        Returns: dict con recompensas para challenger y reasoner
        """
        # Recompensa reasoner basada en precisión
        accuracy = 1.0 - losses.mse_loss(prediction, target)
        reasoner_reward = max(0.0, accuracy)
        
        # Recompensa challenger basada en dificultad balanceada
        # Queremos tareas ni muy fáciles ni muy difíciles
        difficulty = abs(0.5 - accuracy)  # 0 = perfecto, 1 = muy fácil o muy difícil
        challenger_reward = 1.0 - difficulty
        
        # Bonus por diversidad
        diversity_bonus = self.latent_space.get_diversity_reward(task)
        challenger_reward += 0.1 * diversity_bonus
        
        return {
            "reasoner": reasoner_reward,
            "challenger": challenger_reward,
            "difficulty": difficulty,
            "diversity": diversity_bonus
        }
    
    def update_baseline(self, reward: float) -> None:
        """Actualiza baseline para reducir varianza"""
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline + 
            (1 - self.baseline_decay) * reward
        )
    
    def train_step(self, target_fn: Callable[[List[float]], List[float]]) -> Dict[str, float]:
        """
        Un paso completo de entrenamiento RL
        
        Args:
            target_fn: Función que genera el target correcto para una tarea
        """
        # 1. Sample z y generar tarea
        z = self.latent_space.sample()
        task, log_prob = self.generate_task(z)
        
        # 2. Reasoner resuelve
        prediction = self.solve_task(task, z)
        
        # 3. Evaluar
        target = target_fn(task)
        rewards = self.evaluate_task(task, prediction, target)
        
        # 4. Actualizar reasoner (supervisado)
        reasoner_input = task + z
        reasoner_loss = losses.mse_loss(prediction, target)
        
        # Calcular gradiente para reasoner
        dL_dy = losses.mse_grad(prediction, target)
        self.reasoner.train_step(reasoner_input, target, lambda out, tar: dL_dy, self.lr_reasoner)
        
        # 5. Actualizar challenger (REINFORCE)
        challenger_reward = rewards["challenger"]
        self.update_baseline(challenger_reward)
        
        # REINFORCE: gradiente = -log_prob * (reward - baseline)
        advantage = challenger_reward - self.reward_baseline
        
        # Simplificado: actualizar challenger usando la ventaja
        # En producción usarías una política más sofisticada
        challenger_input = z
        challenger_target = [x + 0.01 * advantage * x for x in task]  # Dirección de mejora
        
        dL_dy_challenger = losses.mse_grad(task, challenger_target)
        self.challenger.train_step(challenger_input, challenger_target, 
                                  lambda out, tar: dL_dy_challenger, self.lr_challenger)
        
        # 6. Registrar en espacio latente
        self.latent_space.record(z, challenger_reward, str(self.episode_count))
        
        # 7. Guardar estadísticas
        stats = {
            "episode": self.episode_count,
            "reasoner_reward": rewards["reasoner"],
            "challenger_reward": rewards["challenger"],
            "difficulty": rewards["difficulty"],
            "diversity": rewards["diversity"],
            "reasoner_loss": reasoner_loss,
            "advantage": advantage
        }
        
        self.task_history.append(stats)
        self.episode_count += 1
        
        return stats
    
    def train(self, target_fn: Callable[[List[float]], List[float]], 
              episodes: int = 1000, verbose: bool = True) -> List[Dict[str, float]]:
        """Entrena por múltiples episodios"""
        history = []
        
        for episode in range(episodes):
            stats = self.train_step(target_fn)
            history.append(stats)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = sum(h["reasoner_reward"] for h in history[-100:]) / 100
                print(f"Episode {episode + 1}: Avg reward = {avg_reward:.3f}, "
                      f"Difficulty = {stats['difficulty']:.3f}")
        
        return history
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del curriculum"""
        if not self.task_history:
            return {}
        
        recent = self.task_history[-100:] if len(self.task_history) > 100 else self.task_history
        
        return {
            "total_episodes": self.episode_count,
            "avg_reasoner_reward": sum(h["reasoner_reward"] for h in recent) / len(recent),
            "avg_challenger_reward": sum(h["challenger_reward"] for h in recent) / len(recent),
            "avg_difficulty": sum(h["difficulty"] for h in recent) / len(recent),
            "best_z_vectors": self.latent_space.get_best_z(3)
        }
