#!/usr/bin/env python3
"""
Test funcional de Fase 5 - Variable Latente z.
Solución al error de dimensiones.
"""

import sys
import os
import random

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.network import NeuralNetwork
from core.latent import sample_gaussian_z, LatentProjector
from core import losses


def test_fase5_simple():
    """Test funcional simple de Fase 5."""
    print("🧠 Test Fase 5 - Variable Latente z")
    print("=" * 50)
    
    base_sizes = [2, 4, 1]
    z_dim = 2
    z_proj_dim = 1

    print("📊 Configuración:")
    print(f"   layer_sizes: {base_sizes}")
    print(f"   z_dim: {z_dim}")
    print(f"   z_proj_dim: {z_proj_dim}")

    z_config = {
        "z_dim": z_dim,
        "decoder_layer_idx": 1,  # la capa oculta recibe z
        "z_proj_dim": z_proj_dim
    }

    nn = NeuralNetwork(base_sizes, activation="sigmoid", z_config=z_config)

    print("\n📋 Estructura de la red:")
    expected_inputs = []
    for i, layer in enumerate(nn.layers):
        print(f"   Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
        expected = base_sizes[i]
        if i == z_config["decoder_layer_idx"]:
            expected += z_proj_dim
        expected_inputs.append(expected)
    assert [layer.n_inputs for layer in nn.layers] == expected_inputs
    
    # Test 1: Forward sin z
    x = [0.5, 0.5]
    out_normal = nn.forward(x)
    print(f"\n✅ Forward sin z: {out_normal}")
    
    # Test 2: Forward con z
    z = [0.1, 0.2]
    out_z = nn.forward(x, z=z)
    print(f"✅ Forward con z: {out_z}")
    
    # Test 3: Verificar efecto de z
    z1 = [0.1, 0.2]
    z2 = [-0.1, -0.2]
    
    out1 = nn.forward(x, z=z1)
    out2 = nn.forward(x, z=z2)
    
    diff = abs(out1[0] - out2[0])
    print(f"\n📊 Efecto de z:\n   z1={z1} -> {out1[0]:.4f}\n   z2={z2} -> {out2[0]:.4f}\n   Diferencia: {diff:.4f}")
    assert diff > 5e-5, "Variable latente tiene poco efecto"


def test_entrenamiento_simple():
    """Test de entrenamiento básico con z."""
    print("\n🎯 Test de entrenamiento con z...")
    
    # Configuración nativa con z entrando en la primera capa
    base_sizes = [2, 4, 1]
    z_config = {"z_dim": 1, "decoder_layer_idx": 0, "z_proj_dim": 1}
    nn = NeuralNetwork(base_sizes, activation="sigmoid", z_config=z_config)
    
    # Datos XOR simples
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    
    # Entrenamiento simple
    for epoch in range(100):
        total_loss = 0
        for x, y in dataset:
            z = [random.uniform(-0.5, 0.5)]  # z aleatorio
            
            # Forward
            pred = nn.forward(x, z=z)
            loss = (pred[0] - y[0]) ** 2
            total_loss += loss
            
            # Backprop simple (gradiente manual)
            error = pred[0] - y[0]
            dL_dy = [2 * error]
            nn.train_step(x, y, lambda out, tar: dL_dy, lr=0.1, z=z)
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"   Epoch {epoch}: loss={avg_loss:.4f}")
    
    print("✅ Entrenamiento completado sin errores")
