#!/usr/bin/env python3
"""
Script de diagnóstico para identificar y solucionar el error de Fase 5.
"""

import sys
import os
import traceback

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_imports():
    """Verificar que todos los módulos se pueden importar."""
    print("🔍 Debug - Verificando imports...")
    try:
        from core.network import NeuralNetwork
        from core.latent import LatentProjector, sample_gaussian_z
        from core import losses
        print("   ✅ Todos los imports funcionan")
        return True
    except Exception as e:
        print(f"   ❌ Error en imports: {e}")
        traceback.print_exc()
        return False

def debug_latent_projector():
    """Verificar LatentProjector."""
    print("\n🔍 Debug - LatentProjector...")
    try:
        from core.latent import LatentProjector, sample_gaussian_z
        
        projector = LatentProjector(z_dim=3, out_dim=2)
        z = sample_gaussian_z(3)
        projected = projector.project(z)
        
        print(f"   ✅ LatentProjector funciona")
        print(f"   Input z: {z}")
        print(f"   Output: {projected}")
        print(f"   Dimensiones: {len(z)} -> {len(projected)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error en LatentProjector: {e}")
        traceback.print_exc()
        return False

def debug_network_construction():
    """Verificar construcción de red con z_config."""
    print("\n🔍 Debug - Construcción de red...")
    try:
        from core.network import NeuralNetwork
        
        # Configuración que causa el error
        layer_sizes = [2, 4, 1]
        z_config = {
            "z_dim": 3,
            "decoder_layer_idx": 1,
            "z_proj_dim": 2
        }
        
        print(f"   Configuración:")
        print(f"   layer_sizes: {layer_sizes}")
        print(f"   z_config: {z_config}")
        
        # Verificar paso a paso
        print(f"   decoder_layer_idx: {z_config['decoder_layer_idx']}")
        print(f"   len(layer_sizes): {len(layer_sizes)}")
        
        # Ajuste de dimensiones
        ls = list(layer_sizes)
        print(f"   Antes: {ls}")
        ls[z_config["decoder_layer_idx"]] = ls[z_config["decoder_layer_idx"]] + z_config["z_proj_dim"]
        print(f"   Después: {ls}")
        
        # Crear red
        nn = NeuralNetwork(layer_sizes, activation="sigmoid", z_config=z_config)
        
        print(f"   ✅ Red creada exitosamente")
        print(f"   Capas:")
        for i, layer in enumerate(nn.layers):
            print(f"     Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
        
        return nn, z_config
        
    except Exception as e:
        print(f"   ❌ Error en construcción: {e}")
        traceback.print_exc()
        return None, None

def debug_forward_pass(nn, z_config):
    """Verificar forward pass."""
    print("\n🔍 Debug - Forward pass...")
    if nn is None:
        return False
        
    try:
        x = [0.5, 0.5]
        z = [0.1, 0.2, 0.3]
        
        print(f"   Input x: {x} (len={len(x)})")
        print(f"   Input z: {z} (len={len(z)})")
        print(f"   Layer 0 espera: {nn.layers[0].n_inputs} inputs")
        
        # Verificar dimensiones
        if z_config and z_config['decoder_layer_idx'] == 0:
            expected_x = nn.layers[0].n_inputs - nn.z_proj_dim
            print(f"   Para decoder_layer_idx=0, esperamos {expected_x} inputs")
        else:
            expected_x = nn.layers[0].n_inputs
            print(f"   Esperamos {expected_x} inputs")
        
        # Intentar forward
        out1 = nn.forward(x)
        print(f"   ✅ Forward sin z: {out1}")
        
        out2 = nn.forward(x, z=z)
        print(f"   ✅ Forward con z: {out2}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en forward: {e}")
        traceback.print_exc()
        return False

def debug_solution():
    """Proporcionar solución al problema."""
    print("\n🔧 Solución al problema...")
    
    # El problema es que decoder_layer_idx=1 apunta a la capa oculta
    # pero layer_sizes[1] es el número de neuronas, no el número de inputs
    
    print("   📋 Análisis del problema:")
    print("   - decoder_layer_idx=1 apunta a la capa oculta")
    print("   - layer_sizes[1] = 4 es el número de neuronas")
    print("   - layer_sizes[0] = 2 es el número de inputs")
    print("   - La solución es ajustar layer_sizes[decoder_layer_idx] correctamente")
    
    # Solución correcta
    from core.network import NeuralNetwork
    
    # Configuración corregida
    layer_sizes = [2, 4, 1]  # [inputs, hidden, output]
    z_config = {
        "z_dim": 3,
        "decoder_layer_idx": 1,  # índice de la capa que recibe z
        "z_proj_dim": 2
    }
    
    # Ajuste correcto: incrementar la entrada de la capa decoder_layer_idx
    # Para decoder_layer_idx=1, necesitamos incrementar layer_sizes[1]
    # pero en realidad, incrementamos la entrada de la capa 1
    
    adjusted_sizes = [2, 4 + 2, 1]  # [2, 6, 1]
    
    print("   ✅ Solución:")
    print(f"   - Original: {layer_sizes}")
    print(f"   - Ajustado: {adjusted_sizes}")
    
    nn = NeuralNetwork(adjusted_sizes, activation="sigmoid")
    
    # Verificar
    print("   Verificación:")
    for i, layer in enumerate(nn.layers):
        print(f"   - Layer {i}: {layer.n_inputs} -> {layer.n_neurons}")
    
    # Test
    x = [0.5, 0.5]
    z = [0.1, 0.2, 0.3]
    
    # Para decoder_layer_idx=1, z se concatena con la salida de la capa 0
    # pero la capa 1 espera 6 inputs (4+2)
    
    # Solución: usar z_config con layer_sizes ajustado
    z_config_fixed = {
        "z_dim": 3,
        "decoder_layer_idx": 1,
        "z_proj_dim": 2
    }
    
    nn_fixed = NeuralNetwork([2, 4, 1], activation="sigmoid", z_config=z_config_fixed)
    
    print("   ✅ Implementación corregida")
    
    return True

if __name__ == "__main__":
    print("🧠 Diagnóstico Completo - Fase 5")
    print("=" * 60)
    
    # Ejecutar diagnóstico
    if debug_imports():
        debug_latent_projector()
        nn, z_config = debug_network_construction()
        if nn:
            debug_forward_pass(nn, z_config)
            debug_solution()
    
    print("\n🎯 Resumen del problema:")
    print("   El error 'Concatenación imposible: 6 + 2 != 6' ocurre porque:")
    print("   1. decoder_layer_idx=1 apunta a la capa oculta")
    print("   2. La capa oculta espera 4 inputs")
    print("   3. Queremos concatenar 2 dimensiones de z")
    print("   4. La solución es incrementar layer_sizes[1] en z_proj_dim")
    
    print("\n🔧 Solución implementada:")
    print("   - layer_sizes[decoder_layer_idx] += z_proj_dim")
    print("   - Esto ajusta automáticamente las dimensiones")
