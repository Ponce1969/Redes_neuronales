#!/usr/bin/env python3
"""
Ejemplo de entrenamiento XOR usando autograd Fase 6.
Demuestra el mini-framework en acción.
"""

import sys
import os
import random

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autograd.value import Value
from autograd.functional import linear, mse_loss
from autograd.ops import relu


def main():
    print("🧠 Fase 6 - Entrenamiento XOR con Autograd")
    print("=" * 50)
    
    # Dataset XOR
    data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    
    # Estructura 2 -> 4 -> 1 usando autograd
    def make_layer(n_in, n_out):
        """Crear capa con pesos y biases como Value"""
        layer = []
        for _ in range(n_out):
            weights = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
            bias = Value(random.uniform(-1, 1))
            layer.append((weights, bias))
        return layer
    
    # Capas
    layer1 = make_layer(2, 4)  # 2 -> 4
    layer2 = make_layer(4, 1)  # 4 -> 1
    
    def forward(x):
        """Forward pass usando autograd"""
        vals = [Value(v) for v in x]
        
        # Capa 1 con tanh
        hidden = []
        for weights, bias in layer1:
            h = linear(vals, weights, bias).tanh()
            hidden.append(h)
        
        # Capa 2 con tanh
        out = []
        for weights, bias in layer2:
            o = linear(hidden, weights, bias).tanh()
            out.append(o)
        
        return out
    
    # Entrenamiento
    lr = 0.1
    epochs = 2000
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for x, y in data:
            # Forward
            y_pred = forward(x)
            loss = mse_loss(y_pred, y)
            
            # Reset gradientes
            for layer in (layer1 + layer2):
                for w in layer[0]:
                    w.grad = 0.0
                layer[1].grad = 0.0
            
            # Backward automático
            loss.backward()
            
            # Update
            for layer in (layer1 + layer2):
                for w in layer[0]:
                    w.data -= lr * w.grad
                layer[1].data -= lr * layer[1].grad
            
            total_loss += loss.data
        
        if epoch % 400 == 0:
            avg_loss = total_loss / len(data)
            print(f"   Epoch {epoch:4d} | Loss={avg_loss:.6f}")
    
    # Resultados finales
    print("\n📊 Predicciones finales:")
    for x, y in data:
        y_pred = forward(x)
        print(f"   x={x} -> pred={y_pred[0].data:.4f} (target={y[0]})")
    
    print("\n🎉 ¡Entrenamiento completado con autograd!")
    print("✅ Fase 6 implementada exitosamente")


if __name__ == "__main__":
    main()
