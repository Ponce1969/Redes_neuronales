#!/usr/bin/env python3
"""
Demostración práctica de un agente cognitivo que aprende tareas de predicción secuencial.
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.autograd.value import Value
from src.core.cognitive_block import CognitiveBlock


def mse_loss(predictions, targets):
    """Mean Squared Error Loss"""
    loss = Value(0.0)
    for pred, target in zip(predictions, targets):
        diff = pred - target
        loss = loss + (diff * diff)
    return loss / len(predictions)


def target(seq, t):
    """Función objetivo: predecir si el siguiente número es mayor"""
    if t == len(seq) - 1:
        return [Value(0.0)]
    return [Value(1.0 if seq[t + 1] > seq[t] else 0.0)]


def main():
    # Dataset secuencial (aprende a predecir si el siguiente número será mayor)
    sequences = [
        [0.1, 0.4, 0.2, 0.8, 0.6],
        [0.9, 0.7, 0.5, 0.2, 0.1],
    ]

    print("🧠 Demostración CognitiveBlock - Predicción Secuencial")
    print("=" * 50)
    
    # Crear bloque cognitivo
    block = CognitiveBlock(n_inputs=1, n_hidden=3, n_outputs=1)
    lr = 0.05

    # Entrenamiento
    for epoch in range(1500):
        total_loss = 0.0
        for seq in sequences:
            for t in range(len(seq)):
                x = [Value(seq[t])]
                y_true = target(seq, t)
                y_pred = block.forward(x)
                loss = mse_loss(y_pred, y_true)
                loss.backward()
                
                # Actualización manual de todos los pesos
                params = []
                params += block.perceiver.input_weights
                params += block.perceiver.memory_weights
                params += block.perceiver.gate_weights + [block.perceiver.gate_bias]
                params += [block.perceiver.bias]
                params += block.reasoner.weights_in + block.reasoner.weights_mem
                params += [block.reasoner.bias]
                params += block.decision_weights + [block.decision_bias]
                
                for p in params:
                    p.data -= lr * p.grad
                    p.grad = 0.0
                total_loss += loss.data
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | Loss={total_loss:.6f}")

    # Resultados finales
    print("\n" + "=" * 50)
    print("🎯 Predicción del CognitiveBlock:")
    for seq in sequences:
        print(f"Secuencia: {seq}")
        for t in range(len(seq)):
            x = [Value(seq[t])]
            y_pred = block.forward(x)
            trend = "↑" if y_pred[0].data > 0.5 else "↓"
            print(f"  Input {seq[t]:.2f} → Pred {y_pred[0].data:.3f} {trend}")
        print()

    print("✅ Demostración completada exitosamente!")


if __name__ == "__main__":
    main()
