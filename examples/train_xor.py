#!/usr/bin/env python3
"""
Entrena la red para resolver XOR y muestra resultados.
Este es un ejemplo práctico de la Fase 3 - Aprendizaje y retropropagación.
"""

import sys
import os

# Añadir el directorio src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import NeuralNetwork
from core import losses
from engine.trainer import Trainer

def main():
    print("🧠 Neural Core - Test XOR")
    print("=" * 40)
    
    # Dataset XOR (entrada 2 -> salida 1)
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    # Red: 2 inputs -> 4 hidden -> 1 output
    nn = NeuralNetwork([2, 4, 1], activation="sigmoid")
    trainer = Trainer(nn, loss_fn=losses.mse_loss, loss_grad_fn=losses.mse_grad, lr=0.5, batch_size=1)

    print("=== Antes del entrenamiento ===")
    for x, y in dataset:
        pred = nn.forward(x)
        print(f"{x} -> pred={pred[0]:.4f} target={y[0]}")

    print("\n=== Entrenando... ===")
    trainer.train(dataset, epochs=5000, verbose=False)

    print("\n=== Después del entrenamiento ===")
    for x, y in dataset:
        pred = nn.forward(x)
        print(f"{x} -> pred={pred[0]:.4f} target={y[0]}")

    avg_loss, correct = trainer.evaluate(dataset)
    print(f"\n📊 Métricas finales:")
    print(f"   Loss final: {avg_loss:.6f}")
    print(f"   Precisión: {correct}/{len(dataset)} ({100*correct/len(dataset):.1f}%)")
    
    # Umbral de decisión para clasificación binaria
    print(f"\n🔍 Predicciones con umbral 0.5:")
    for x, y in dataset:
        pred = nn.forward(x)
        decision = 1 if pred[0] >= 0.5 else 0
        correct_str = "✅" if decision == int(y[0]) else "❌"
        print(f"   {x} -> {pred[0]:.4f} -> {decision} (esperado: {int(y[0])}) {correct_str}")

if __name__ == "__main__":
    main()
