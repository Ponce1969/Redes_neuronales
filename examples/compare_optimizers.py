#!/usr/bin/env python3
"""
Ejemplo de comparación de optimizadores para la Fase 4.
Compara SGD, Momentum y Adam en el problema XOR.
"""

import sys
import os
import time

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import NeuralNetwork
from core import losses
from core.optimizers import SGD, SGDMomentum, Adam
from engine.trainer import Trainer


def create_xor_dataset():
    """Crea el dataset XOR"""
    return [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]


def train_with_optimizer(optimizer_name, optimizer, epochs=2000):
    """Entrena la red con un optimizador específico"""
    print(f"\n🧪 Probando {optimizer_name}...")
    
    # Crear red
    nn = NeuralNetwork([2, 4, 1], activation="sigmoid")
    
    # Configurar trainer con el optimizador
    trainer = Trainer(
        nn, 
        loss_fn=losses.mse_loss, 
        loss_grad_fn=losses.mse_grad,
        optimizer=optimizer,
        lr=0.5 if optimizer_name == "SGD" else 0.1,
        batch_size=1
    )
    
    dataset = create_xor_dataset()
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    trainer.train(dataset, epochs=epochs, verbose=False)
    training_time = time.time() - start_time
    
    # Evaluar
    avg_loss, correct = trainer.evaluate(dataset)
    
    # Mostrar predicciones
    print(f"   📊 Resultados {optimizer_name}:")
    print(f"      Loss final: {avg_loss:.6f}")
    print(f"      Precisión: {correct}/4 ({100*correct/4:.1f}%)")
    print(f"      Tiempo: {training_time:.2f}s")
    
    # Predicciones detalladas
    print(f"      Predicciones:")
    for x, y in dataset:
        pred = nn.forward(x)
        print(f"         {x} -> {pred[0]:.4f} (target: {y[0]})")
    
    return {
        "optimizer": optimizer_name,
        "loss": avg_loss,
        "accuracy": correct/4,
        "time": training_time
    }


def main():
    print("🧠 Neural Core - Fase 4: Comparación de Optimizadores")
    print("=" * 60)
    
    # Configuración
    epochs = 2000
    
    # Optimizadores a probar
    optimizers = [
        ("SGD", SGD(lr=0.5)),
        ("Momentum", SGDMomentum(lr=0.1, momentum=0.9)),
        ("Adam", Adam(lr=0.1)),
    ]
    
    results = []
    
    for name, optimizer in optimizers:
        result = train_with_optimizer(name, optimizer, epochs)
        results.append(result)
    
    # Resumen comparativo
    print(f"\n📈 Resumen comparativo ({epochs} épocas):")
    print("-" * 60)
    print(f"{'Optimizador':<12} {'Loss':<10} {'Accuracy':<10} {'Tiempo (s)':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['optimizer']:<12} {result['loss']:<10.6f} {result['accuracy']:<10.2%} {result['time']:<10.2f}")
    
    # Recomendación
    best = min(results, key=lambda x: x['loss'])
    print(f"\n🏆 Mejor optimizador: {best['optimizer']} (loss: {best['loss']:.6f})")
    
    # Ejemplo de uso práctico
    print(f"\n💡 Ejemplo de uso:")
    print("""
    # Usar Adam en lugar de SGD:
    from core.optimizers import Adam
    
    trainer = Trainer(
        network=nn,
        loss_fn=losses.mse_loss,
        loss_grad_fn=losses.mse_grad,
        optimizer=Adam(lr=0.01),  # Cambiar optimizador
        batch_size=1
    )
    """)


if __name__ == "__main__":
    main()
