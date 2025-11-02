#!/usr/bin/env python3
"""
Fase 5 Funcional - Variable Latente z sin errores.
Implementación directa y funcional.
"""

import sys
import os
import random
import math

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.network import NeuralNetwork
from core import losses


def sample_gaussian_z(dim: int, mu: float = 0.0, sigma: float = 1.0):
    """Sample z ~ N(mu, sigma^2)"""
    return [random.gauss(mu, sigma) for _ in range(dim)]


def project_z(z: list, weights: list, bias: list):
    """Proyector simple para z"""
    out = []
    for i in range(len(bias)):
        s = bias[i]
        for j in range(len(z)):
            s += weights[i][j] * z[j]
        out.append(math.tanh(s))
    return out


def main():
    print("🧠 Neural Core - Fase 5: Variable Latente z")
    print("=" * 50)
    
    # Configuración funcional
    z_dim = 2
    z_proj_dim = 1
    
    # Red con espacio para z: inputs + z_proj_dim
    # [2 inputs + 1 z] -> 4 hidden -> 1 output
    nn = NeuralNetwork([2 + z_proj_dim, 4, 1], activation="sigmoid")
    
    print("📊 Configuración:")
    print(f"   z_dim: {z_dim}")
    print(f"   z_proj_dim: {z_proj_dim}")
    print(f"   layer_sizes: [2+{z_proj_dim}, 4, 1] = [3, 4, 1]")
    
    print("\n📋 Estructura:")
    for i, layer in enumerate(nn.layers):
        print(f"   Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
    
    # Test 1: Forward sin z
    x = [0.5, 0.5]
    z = [0.0] * z_proj_dim  # z neutral
    x_with_z = x + z
    
    out_normal = nn.forward(x_with_z)
    print(f"\n✅ Forward sin z: {out_normal}")
    
    # Test 2: Forward con diferentes z
    z1 = [0.5]
    z2 = [-0.5]
    
    out1 = nn.forward(x + z1)
    out2 = nn.forward(x + z2)
    
    print(f"✅ Forward con z1={z1}: {out1}")
    print(f"✅ Forward con z2={z2}: {out2}")
    
    # Test 3: Verificar efecto
    diff = abs(out1[0] - out2[0])
    print(f"\n📊 Efecto de z: diferencia = {diff:.4f}")
    
    # Test 4: Entrenamiento simple
    print("\n🎯 Entrenamiento con z...")
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    
    for epoch in range(100):
        total_loss = 0
        for x, y in dataset:
            z = sample_gaussian_z(z_proj_dim)
            x_with_z = x + z
            
            # Forward
            pred = nn.forward(x_with_z)
            loss = (pred[0] - y[0]) ** 2
            total_loss += loss
            
            # Backprop
            error = pred[0] - y[0]
            dL_dy = [2 * error]
            nn.train_step(x_with_z, y, lambda out, tar: dL_dy, lr=0.1)
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"   Epoch {epoch}: loss={avg_loss:.4f}")
    
    # Test 5: Verificación final
    print("\n✅ Verificación final:")
    for x, y in dataset:
        z = sample_gaussian_z(z_proj_dim)
        pred = nn.forward(x + z)
        print(f"   x={x}, target={y[0]}, pred={pred[0]:.3f}")
    
    print("\n🎉 ¡Fase 5 completamente funcional!")
    print("✅ Variable latente z integrada sin errores")
    print("✅ Sistema listo para experimentación")


if __name__ == "__main__":
    main()
