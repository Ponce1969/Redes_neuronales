"""
Test de integración para variable latente z.
Verifica que la red cambia su salida cuando se varía z y que el sistema funciona correctamente.
"""

import sys
import os
import math

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import NeuralNetwork
from core.latent import sample_gaussian_z, LatentProjector
from core import losses


def test_projector_and_forward_change():
    """
    Test principal: verifica que la salida cambia al variar z.
    """
    print("🔍 Test de integración de variable latente z...")
    
    # Configuración de red con z
    base_sizes = [2, 4, 1]
    z_dim = 3
    z_proj_dim = 2
    decoder_layer_idx = 1  # concatenar antes de la capa oculta->salida
    
    nn = NeuralNetwork(
        layer_sizes=base_sizes,
        activation="sigmoid",
        z_config={
            "z_dim": z_dim,
            "decoder_layer_idx": decoder_layer_idx,
            "z_proj_dim": z_proj_dim
        }
    )
    
    # Verificar configuración
    assert nn.z_dim == z_dim
    assert nn.decoder_layer_idx == decoder_layer_idx
    assert nn.z_proj_dim == z_proj_dim
    assert nn.projector is not None
    
    # Datos de test
    x = [0.1, 0.9]
    
    # Generar diferentes valores de z
    z1 = sample_gaussian_z(z_dim)
    z2 = sample_gaussian_z(z_dim)
    z3 = sample_gaussian_z(z_dim)
    
    # Calcular predicciones
    out1 = nn.forward(x, z=z1)
    out2 = nn.forward(x, z=z2)
    out3 = nn.forward(x, z=z3)
    
    # Verificaciones básicas
    assert isinstance(out1, list)
    assert isinstance(out2, list)
    assert isinstance(out3, list)
    assert len(out1) == len(out2) == len(out3) == 1
    assert 0 <= out1[0] <= 1
    assert 0 <= out2[0] <= 1
    assert 0 <= out3[0] <= 1
    
    # Verificar que las salidas son diferentes (con alta probabilidad)
    # Si todas son iguales, algo está mal
    predictions = [out1[0], out2[0], out3[0]]
    variance = sum((p - sum(predictions)/3) ** 2 for p in predictions) / 3
    
    print(f"   Predicciones para x={x}:")
    print(f"   z1 -> {out1[0]:.4f}")
    print(f"   z2 -> {out2[0]:.4f}")
    print(f"   z3 -> {out3[0]:.4f}")
    print(f"   Varianza: {variance:.6f}")
    
    # La varianza debe ser significativa (no todas iguales)
    assert variance > 1e-6, "Variable latente z no está afectando la salida"
    
    print("   ✅ Variable latente z está funcionando correctamente")


def test_projector_dimensions():
    """Test de dimensiones del projector."""
    print("🔍 Test de dimensiones del projector...")
    
    z_dim = 4
    out_dim = 2
    projector = LatentProjector(z_dim, out_dim)
    
    # Verificar dimensiones
    assert len(projector.weights) == out_dim
    assert len(projector.weights[0]) == z_dim
    assert len(projector.bias) == out_dim
    
    # Test de proyección
    z = sample_gaussian_z(z_dim)
    projected = projector.project(z)
    
    assert len(projected) == out_dim
    assert all(-1 <= val <= 1 for val in projected)  # tanh output
    
    print("   ✅ Projector tiene dimensiones correctas")


def test_network_compatibility():
    """Test de compatibilidad con redes sin z."""
    print("🔍 Test de compatibilidad sin z...")
    
    # Red sin z
    nn = NeuralNetwork([2, 3, 1], activation="sigmoid")
    
    x = [0.5, 0.5]
    y = nn.forward(x)
    y_with_none = nn.forward(x, z=None)
    
    # Deben ser iguales cuando z=None
    assert y == y_with_none
    assert nn.z_dim == 0
    assert nn.projector is None
    
    print("   ✅ Compatibilidad retroactiva mantenida")


def test_network_with_z_config():
    """Test de configuración completa con z."""
    print("🔍 Test de configuración con z...")
    
    # Configuración completa
    z_config = {
        "z_dim": 5,
        "decoder_layer_idx": 1,
        "z_proj_dim": 3
    }
    
    nn = NeuralNetwork([3, 4, 2, 1], activation="sigmoid", z_config=z_config)
    
    # Verificar ajuste de dimensiones
    expected_layer_sizes = [3, 4 + 3, 2, 1]  # capa 1 incrementada en z_proj_dim
    
    actual_sizes = [layer.n_inputs for layer in nn.layers]
    expected_inputs = expected_layer_sizes[:-1]  # inputs de cada capa
    
    assert actual_sizes == expected_inputs
    
    # Test de forward
    x = [1.0, 0.5, -0.5]
    z = sample_gaussian_z(5)
    
    output = nn.forward(x, z=z)
    assert len(output) == 1
    assert 0 <= output[0] <= 1
    
    print("   ✅ Configuración con z funciona correctamente")


def test_train_step_with_z():
    """Test de entrenamiento con z."""
    print("🔍 Test de entrenamiento con z...")
    
    # Red pequeña para test rápido
    z_config = {"z_dim": 2, "decoder_layer_idx": 0, "z_proj_dim": 1}
    nn = NeuralNetwork([2, 1], activation="sigmoid", z_config=z_config)
    
    x = [0.5, 0.3]
    y = [1.0]
    z = sample_gaussian_z(2)
    
    # Entrenar un paso sin errores
    try:
        loss_before = losses.mse_loss(nn.forward(x, z=z), y)
        nn.train_step(x, y, losses.mse_grad, lr=0.01, z=z)
        loss_after = losses.mse_loss(nn.forward(x, z=z), y)
        
        assert math.isfinite(loss_after)
        print(f"   ✅ Entrenamiento con z: loss {loss_before:.4f} -> {loss_after:.4f}")
        
    except Exception as e:
        raise AssertionError(f"Error en entrenamiento con z: {e}")


if __name__ == "__main__":
    print("🧠 Neural Core - Fase 5: Tests de Integración")
    print("=" * 50)
    
    test_projector_and_forward_change()
    test_projector_dimensions()
    test_network_compatibility()
    test_network_with_z_config()
    test_train_step_with_z()
    
    print("\n🎉 ¡Todos los tests de integración pasaron!")
    print("✅ Variable latente z completamente integrada")
    print("✅ Sistema listo para auto-curriculum learning")
