#!/usr/bin/env python3
"""
Test funcional final de Fase 5 - Variable Latente z.
Solución completa al error de dimensiones.
"""

import sys
import os
import random

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.network import NeuralNetwork
from core.latent import LatentProjector
from core import losses


def test_fase5_corregido():
    """Test de Fase 5 con solución de dimensiones."""
    print("🧠 Fase 5 - Variable Latente z (Corregido)")
    print("=" * 50)
    
    # Configuración nativa usando z_config
    base_sizes = [2, 4, 1]
    z_dim = 2
    z_proj_dim = 1
    z_config = {
        "z_dim": z_dim,
        "decoder_layer_idx": 1,
        "z_proj_dim": z_proj_dim,
    }

    print("📊 Configuración de dimensiones:")
    print(f"   Base: {base_sizes}")
    print(f"   z_dim: {z_dim}")
    print(f"   z_proj_dim: {z_proj_dim}")

    nn = NeuralNetwork(base_sizes, activation="sigmoid", z_config=z_config)
    
    print("\n📋 Estructura de capas:")
    expected_inputs = []
    for i, layer in enumerate(nn.layers):
        print(f"   Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
        n_inputs = base_sizes[i]
        if i == z_config["decoder_layer_idx"]:
            n_inputs += z_proj_dim
        expected_inputs.append(n_inputs)
    assert [layer.n_inputs for layer in nn.layers] == expected_inputs
    
    # Test 1: Forward sin z
    x = [0.5, 0.5]
    try:
        out_normal = nn.forward(x)
        print(f"\n✅ Forward sin z: {out_normal}")
    except Exception as e:
        raise AssertionError(f"Error sin z: {e}")
    
    # Test 2: Forward con z (nativo)
    z = [0.1, -0.2]
    out_z = nn.forward(x, z=z)
    print(f"✅ Forward con z: {out_z}")
    
    # Test 3: Verificar efecto de z
    z1 = [0.1, 0.0]
    z2 = [-0.1, 0.3]

    out1 = nn.forward(x, z=z1)
    out2 = nn.forward(x, z=z2)
    
    diff = abs(out1[0] - out2[0])
    print(f"\n📊 Efecto de z:\n   z1={z1} -> {out1[0]:.4f}\n   z2={z2} -> {out2[0]:.4f}\n   Diferencia: {diff:.4f}")
    assert diff > 1e-4, "Variable latente z tiene poco efecto"


def test_con_z_config():
    """Test con z_config implementado correctamente."""
    print("\n🎯 Test con z_config correcto...")
    
    # Configuración con projector explícito para verificar compatibilidad
    z_config = {
        "z_dim": 2,
        "decoder_layer_idx": 1,
        "z_proj_dim": 1,
        "projector": LatentProjector(2, 1),
    }

    base_sizes = [2, 4, 1]
    nn = NeuralNetwork(base_sizes, activation="sigmoid", z_config=z_config)

    print("Configuración con z:")
    print(f"   layer_sizes: {base_sizes}")
    print(f"   z_dim: {z_config['z_dim']}")
    print(f"   decoder_layer_idx: {z_config['decoder_layer_idx']}")
    print(f"   z_proj_dim: {z_config['z_proj_dim']}")

    x = [0.5, 0.5]
    z = [0.1, 0.2]
    out = nn.forward(x, z=z)
    print(f"✅ Forward exitoso: {out}")

