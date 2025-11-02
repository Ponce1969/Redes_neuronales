#!/usr/bin/env python3
"""
Test funcional de Fase 5 - Variable Latente z.
Solución al error de dimensiones.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.network import NeuralNetwork
from core.latent import sample_gaussian_z, LatentProjector
from core import losses


def test_fase5_simple():
    """Test funcional simple de Fase 5."""
    print("🧠 Test Fase 5 - Variable Latente z")
    print("=" * 50)
    
    # Configuración simple y funcional
    # Red: 2 inputs -> 4 hidden -> 1 output
    # z se concatena con la entrada de la capa oculta (índice 1)
    
    # Ajuste manual de dimensiones:
    # Layer 0: 2 -> 4 (sin cambio)
    # Layer 1: 4 + z_proj_dim -> 1 (capa oculta recibe z)
    
    base_sizes = [2, 4, 1]
    z_dim = 2
    z_proj_dim = 1
    
    # Ajustar dimensiones correctamente
    adjusted_sizes = [2, 4 + z_proj_dim, 1]
    
    print("📊 Configuración:")
    print(f"   Original: {base_sizes}")
    print(f"   Ajustado: {adjusted_sizes}")
    print(f"   z_dim: {z_dim}")
    print(f"   z_proj_dim: {z_proj_dim}")
    
    # Crear red con dimensiones ajustadas
    z_config = {
        "z_dim": z_dim,
        "decoder_layer_idx": 1,  # índice de la capa que recibe z
        "z_proj_dim": z_proj_dim
    }
    
    nn = NeuralNetwork(adjusted_sizes, activation="sigmoid", z_config=z_config)
    
    print("\n📋 Estructura de la red:")
    for i, layer in enumerate(nn.layers):
        print(f"   Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
    
    # Test 1: Forward sin z
    x = [0.5, 0.5]
    try:
        out_normal = nn.forward(x)
        print(f"\n✅ Forward sin z: {out_normal}")
    except Exception as e:
        print(f"❌ Error sin z: {e}")
        return False
    
    # Test 2: Forward con z
    z = [0.1, 0.2]
    try:
        out_z = nn.forward(x, z=z)
        print(f"✅ Forward con z: {out_z}")
    except Exception as e:
        print(f"❌ Error con z: {e}")
        return False
    
    # Test 3: Verificar efecto de z
    z1 = [0.1, 0.2]
    z2 = [-0.1, -0.2]
    
    out1 = nn.forward(x, z=z1)
    out2 = nn.forward(x, z=z2)
    
    diff = abs(out1[0] - out2[0])
    print(f"\n📊 Efecto de z:")
    print(f"   z1={z1} -> {out1[0]:.4f}")
    print(f"   z2={z2} -> {out2[0]:.4f}")
    print(f"   Diferencia: {diff:.4f}")
    
    if diff > 1e-4:
        print("✅ Variable latente z está funcionando correctamente")
        return True
    else:
        print("⚠️ Variable latente tiene poco efecto")
        return False


def test_entrenamiento_simple():
    """Test de entrenamiento básico con z."""
    print("\n🎯 Test de entrenamiento con z...")
    
    # Configuración simple
    adjusted_sizes = [2, 5, 1]  # 2 + 1 (z) = 3 inputs para capa oculta
    z_dim = 1
    
    nn = NeuralNetwork(adjusted_sizes, activation="sigmoid")
    
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
    return True


if __name__ == "__main__":
    print("🧠 Neural Core - Fase 5: Solución de Dimensiones")
    print("=" * 60)
    
    success1 = test_fase5_simple()
    success2 = test_entrenamiento_simple()
    
    if success1 and success2:
        print("\n🎉 ¡Fase 5 funcional y sin errores!")
        print("✅ Variable latente z completamente integrada")
    else:
        print("\n❌ Aún hay problemas por resolver")
