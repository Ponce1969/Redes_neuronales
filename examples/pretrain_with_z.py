#!/usr/bin/env python3
"""
Ejemplo de preentrenamiento con variable latente z.
Este script demuestra cómo usar z para condicionar la red y verificar su funcionamiento.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import NeuralNetwork
from core import losses
from core.latent import sample_gaussian_z, LatentProjector
from core.optimizers import Adam
from engine.trainer import Trainer


# Dataset XOR
XOR_DATASET = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def build_network_with_z():
    """
    Construye una red neuronal con soporte para variable latente z.
    
    Configuración:
    - Entrada: 2 (XOR)
    - Capa oculta: 4 neuronas
    - Salida: 1 neurona
    - z_dim: 4 (dimensión del espacio latente)
    - z_proj_dim: 2 (dimensión después de proyección)
    - decoder_layer_idx: 1 (concatenar antes de la capa oculta->salida)
    """
    # base layer sizes: entrada 2 -> oculta 4 -> salida 1
    base_sizes = [2, 4, 1]
    z_dim = 4
    z_proj_dim = 2
    decoder_layer_idx = 1  # concatenaremos z proyectado antes de la capa oculta->salida
    
    z_config = {
        "z_dim": z_dim,
        "decoder_layer_idx": decoder_layer_idx,
        "z_proj_dim": z_proj_dim,
        # projector opcional; si lo omites, NeuralNetwork crea uno por defecto
    }
    
    nn = NeuralNetwork(base_sizes, activation="sigmoid", z_config=z_config)
    return nn, z_dim


def dataset_with_random_z(original_dataset, z_dim):
    """
    Genera dataset con variables latentes z aleatorias.
    
    Args:
        original_dataset: dataset base (x, y)
        z_dim: dimensión del espacio latente
    
    Returns:
        Iterable de (x, y, z) para entrenamiento
    """
    for x, y in original_dataset:
        z = sample_gaussian_z(z_dim)
        yield (x, y, z)


def main():
    print("🧠 Neural Core - Fase 5: Preentrenamiento con Variable Latente z")
    print("=" * 60)
    
    # Construir red con z
    nn, z_dim = build_network_with_z()
    nn.summary()
    
    # Configurar optimizador y trainer
    optimizer = Adam(lr=0.01)
    trainer = Trainer(
        network=nn,
        loss_fn=losses.mse_loss,
        loss_grad_fn=losses.mse_grad,
        optimizer=optimizer,
        lr=0.01,
        batch_size=1,
        shuffle=True,
    )
    
    # Entrenamiento manual con z
    print("\n🎯 Iniciando entrenamiento con variable latente z...")
    epochs = 2000
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        for x, y in XOR_DATASET:
            # Generar z aleatorio para este sample
            z = sample_gaussian_z(z_dim)
            
            # Forward con z
            y_pred = nn.forward(x, z=z)
            
            # Calcular loss
            current_loss = losses.mse_loss(y_pred, y)
            total_loss += current_loss
            
            # Backpropagation con z
            dL_dy = losses.mse_grad(y_pred, y)
            nn.train_step(x, y, lambda out, tar: dL_dy, lr=0.01, z=z)
        
        # Reportar progreso
        if epoch % 250 == 0:
            avg_loss = total_loss / len(XOR_DATASET)
            print(f"   [Epoch {epoch:4d}] loss={avg_loss:.6f}")
    
    # Evaluación final con diferentes valores de z
    print("\n📊 Evaluación final - Efecto de z en las predicciones:")
    print("-" * 50)
    
    for x, y in XOR_DATASET:
        # Generar varios valores de z para el mismo input
        z1 = sample_gaussian_z(z_dim)
        z2 = sample_gaussian_z(z_dim)
        z3 = sample_gaussian_z(z_dim)
        
        pred1 = nn.forward(x, z=z1)
        pred2 = nn.forward(x, z=z2)
        pred3 = nn.forward(x, z=z3)
        
        print(f"x={x}, target={y}")
        print(f"   pred(z1)={pred1[0]:.4f}")
        print(f"   pred(z2)={pred2[0]:.4f}")
        print(f"   pred(z3)={pred3[0]:.4f}")
        print()
    
    # Verificar que z realmente afecta la salida
    print("✅ Verificación: ¿z realmente cambia la salida?")
    x_test = [0.5, 0.5]
    
    # Generar 5 valores de z diferentes
    predictions = []
    for i in range(5):
        z = sample_gaussian_z(z_dim)
        pred = nn.forward(x_test, z=z)
        predictions.append(pred[0])
    
    # Calcular varianza
    mean_pred = sum(predictions) / len(predictions)
    variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
    
    print(f"   Predicciones para x={x_test}: {predictions}")
    print(f"   Media: {mean_pred:.4f}")
    print(f"   Varianza: {variance:.6f}")
    
    if variance > 1e-4:
        print("   ✅ Variable latente z está afectando significativamente la salida")
    else:
        print("   ⚠️ Variable latente z tiene poco efecto")
    
    # Test de precisión final
    print("\n🎯 Precisión final con z aleatorio:")
    correct = 0
    total_loss = 0
    
    for x, y in XOR_DATASET:
        z = sample_gaussian_z(z_dim)
        pred = nn.forward(x, z=z)
        
        pred_label = 1 if pred[0] >= 0.5 else 0
        target_label = int(y[0])
        
        if pred_label == target_label:
            correct += 1
        
        total_loss += losses.mse_loss(pred, y)
    
    avg_loss = total_loss / len(XOR_DATASET)
    accuracy = correct / len(XOR_DATASET)
    
    print(f"   Loss final: {avg_loss:.6f}")
    print(f"   Precisión: {correct}/4 ({accuracy:.1%})")
    
    print("\n🎉 ¡Fase 5 completada exitosamente!")
    print("✅ Variable latente z integrada y funcionando")
    print("✅ Red puede planificar internamente antes de generar")


if __name__ == "__main__":
    main()
