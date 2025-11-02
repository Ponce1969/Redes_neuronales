"""
Tests de estabilidad del entrenamiento con XOR
para verificar convergencia de loss usando distintos optimizadores.
"""

from core.network import NeuralNetwork
from core import losses
from core.optimizers import SGD, Adam, SGDMomentum, RMSprop
from engine.trainer import Trainer


XOR_DATASET = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def train_and_measure(optimizer, epochs=3000, verbose=False):
    """
    Entrena XOR y mide la pérdida final.
    """
    nn = NeuralNetwork([2, 4, 1], activation="sigmoid")
    
    trainer = Trainer(
        network=nn,
        loss_fn=losses.mse_loss,
        loss_grad_fn=losses.mse_grad,
        optimizer=optimizer,
        lr=getattr(optimizer, "lr", 0.01),
        batch_size=1,
        shuffle=True,
    )
    
    trainer.train(XOR_DATASET, epochs=epochs, verbose=verbose)
    
    # Evaluar error medio final
    total_loss = 0
    for x, y in XOR_DATASET:
        pred = nn.forward(x)
        total_loss += losses.mse_loss(pred, y)
    avg_loss = total_loss / len(XOR_DATASET)
    
    # Contar aciertos
    correct = 0
    for x, y in XOR_DATASET:
        pred = nn.forward(x)
        pred_label = 1 if pred[0] >= 0.5 else 0
        if pred_label == int(y[0]):
            correct += 1
    
    return avg_loss, correct


def test_sgd_converge():
    """Test de convergencia con SGD simple."""
    print("🧪 Test SGD convergencia...")
    optimizer = SGD(lr=0.1)
    loss, correct = train_and_measure(optimizer)
    
    print(f"   SGD: loss={loss:.4f}, accuracy={correct}/4")
    assert loss < 0.05, f"SGD no converge bien (loss={loss})"
    assert correct >= 3, f"SGD precisión baja ({correct}/4)"
    print("   ✅ SGD converge correctamente")


def test_sgd_momentum_converge():
    """Test de convergencia con SGD + Momentum."""
    print("🧪 Test SGD+Momentum convergencia...")
    optimizer = SGDMomentum(lr=0.05, momentum=0.9)
    loss, correct = train_and_measure(optimizer)
    
    print(f"   Momentum: loss={loss:.4f}, accuracy={correct}/4")
    assert loss < 0.05, f"SGD+Momentum no converge bien (loss={loss})"
    assert correct >= 3, f"SGD+Momentum precisión baja ({correct}/4)"
    print("   ✅ Momentum converge correctamente")


def test_adam_converge():
    """Test de convergencia con Adam."""
    print("🧪 Test Adam convergencia...")
    optimizer = Adam(lr=0.01)
    loss, correct = train_and_measure(optimizer)
    
    print(f"   Adam: loss={loss:.4f}, accuracy={correct}/4")
    assert loss < 0.02, f"Adam no converge bien (loss={loss})"
    assert correct >= 3, f"Adam precisión baja ({correct}/4)"
    print("   ✅ Adam converge correctamente")


def test_rmsprop_converge():
    """Test de convergencia con RMSprop."""
    print("🧪 Test RMSprop convergencia...")
    optimizer = RMSprop(lr=0.01)
    loss, correct = train_and_measure(optimizer)
    
    print(f"   RMSprop: loss={loss:.4f}, accuracy={correct}/4")
    assert loss < 0.05, f"RMSprop no converge bien (loss={loss})"
    assert correct >= 3, f"RMSprop precisión baja ({correct}/4)"
    print("   ✅ RMSprop converge correctamente")


def test_stability_comparison():
    """Comparación de estabilidad entre optimizadores."""
    print("📊 Comparación de estabilidad entre optimizadores:")
    
    optimizers = [
        ("SGD", SGD(lr=0.1)),
        ("Momentum", SGDMomentum(lr=0.05, momentum=0.9)),
        ("Adam", Adam(lr=0.01)),
        ("RMSprop", RMSprop(lr=0.01)),
    ]
    
    results = []
    for name, optimizer in optimizers:
        loss, correct = train_and_measure(optimizer, verbose=False)
        results.append((name, loss, correct))
        print(f"   {name:<10}: loss={loss:.4f}, accuracy={correct}/4")
    
    # Verificar que todos convergen
    for name, loss, correct in results:
        assert loss < 0.1, f"{name} no es estable (loss={loss})"
        assert correct >= 3, f"{name} no es preciso ({correct}/4)"
    
    print("   ✅ Todos los optimizadores son estables")


if __name__ == "__main__":
    print("🧠 Neural Core - Fase 4B: Tests de Estabilidad")
    print("=" * 50)
    
    test_sgd_converge()
    test_sgd_momentum_converge()
    test_adam_converge()
    test_rmsprop_converge()
    test_stability_comparison()
    
    print("\n🎉 ¡Todos los tests de estabilidad pasaron!")
    print("✅ Backprop es correcto")
    print("✅ Gradientes son estables")
    print("✅ Optimizadores funcionan")
    print("✅ Motor neuronal listo para Fase 5")
