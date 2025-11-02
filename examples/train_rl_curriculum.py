#!/usr/bin/env python3
"""
Ejemplo de auto-curriculum learning con RL simple.
Este ejemplo demuestra la Fase 4 en acción.
"""

import sys
import os
import math

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import NeuralNetwork
from engine.rl_trainer import SimpleRLTrainer


def simple_math_task(task: List[float]) -> List[float]:
    """
    Función target simple: operaciones matemáticas básicas.
    Genera problemas de dificultad variable.
    """
    # Tarea: [a, b] -> [a + b, a * b]
    # Usamos solo los primeros 2 elementos para simplificar
    a = task[0] if len(task) > 0 else 0.0
    b = task[1] if len(task) > 1 else 0.0
    
    # Normalizar a [-1, 1]
    a = max(min(a, 1.0), -1.0)
    b = max(min(b, 1.0), -1.0)
    
    sum_result = a + b
    prod_result = a * b
    
    # Normalizar resultados a [-1, 1]
    sum_norm = sum_result / 2.0
    prod_norm = prod_result
    
    return [sum_norm, prod_norm]


def main():
    print("🧠 Neural Core - Fase 4: Auto-Curriculum RL")
    print("=" * 50)
    
    # Configuración
    latent_dim = 4
    task_dim = 2
    output_dim = 2
    
    # Crear redes
    challenger = NeuralNetwork([latent_dim, 8, task_dim], activation="tanh")
    reasoner = NeuralNetwork([task_dim + latent_dim, 12, output_dim], activation="tanh")
    
    # Crear trainer
    trainer = SimpleRLTrainer(
        challenger=challenger,
        reasoner=reasoner,
        latent_dim=latent_dim,
        lr_challenger=0.02,
        lr_reasoner=0.01
    )
    
    print("📊 Configuración inicial:")
    print(f"   Dimensión latente: {latent_dim}")
    print(f"   Dimensión tarea: {task_dim}")
    print(f"   Dimensión salida: {output_dim}")
    
    # Entrenar
    print("\n🎯 Iniciando entrenamiento RL...")
    history = trainer.train(simple_math_task, episodes=500, verbose=True)
    
    # Estadísticas finales
    stats = trainer.get_curriculum_stats()
    
    print(f"\n📈 Estadísticas finales:")
    print(f"   Episodios completados: {stats['total_episodes']}")
    print(f"   Recompensa reasoner promedio: {stats['avg_reasoner_reward']:.3f}")
    print(f"   Recompensa challenger promedio: {stats['avg_challenger_reward']:.3f}")
    print(f"   Dificultad promedio: {stats['avg_difficulty']:.3f}")
    
    # Mostrar algunas tareas generadas
    print(f"\n🔍 Ejemplos de tareas generadas:")
    for i in range(3):
        z = trainer.latent_space.sample()
        task, _ = trainer.generate_task(z)
        target = simple_math_task(task)
        prediction = trainer.solve_task(task, z)
        
        print(f"   Tarea: [{task[0]:.3f}, {task[1]:.3f}] ->")
        print(f"   Target: [{target[0]:.3f}, {target[1]:.3f}]")
        print(f"   Predicción: [{prediction[0]:.3f}, {prediction[1]:.3f}]")
        print()


if __name__ == "__main__":
    main()
