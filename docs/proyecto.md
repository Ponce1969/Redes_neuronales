# 🧠 Neural Core - Documentación del Proyecto

## 📋 Descripción General

**Neural Core** es un motor neuronal modular construido completamente en Python puro, diseñado para aprender y experimentar con redes neuronales desde cero, sin frameworks externos.

## 🎯 Objetivos del Proyecto

- **Construir desde cero**: Implementar redes neuronales sin dependencias pesadas
- **Modularidad**: Cada componente es intercambiable y extensible
- **Educación**: Código limpio y bien documentado para aprendizaje
- **Experimentación**: Facilitar pruebas con diferentes arquitecturas y optimizadores

## 📁 Estructura del Proyecto

```
neural_core/
├── src/
│   ├── core/                    # Núcleo neuronal
│   │   ├── neuron.py           # Microneuronas individuales
│   │   ├── layer.py            # Capas de neuronas
│   │   ├── network.py          # Red neuronal completa
│   │   ├── activations.py      # Funciones de activación
│   │   ├── losses.py           # Funciones de pérdida
│   │   ├── optimizers.py       # Optimizadores modulares
│   │   └── latent.py           # Espacio latente (Fase 5)
│   ├── engine/                 # Motor de entrenamiento
│   │   ├── trainer.py          # Entrenamiento supervisado
│   │   └── rl_trainer.py       # Entrenamiento RL (Fase 5)
│   └── app.py                  # Punto de entrada
├── tests/                      # Tests de validación
│   ├── test_gradients.py       # Test de gradientes
│   ├── test_stability.py       # Test de estabilidad
│   └── test_validation.py      # Validación básica
├── examples/                   # Ejemplos prácticos
│   ├── train_xor.py            # XOR con diferentes optimizadores
│   ├── compare_optimizers.py   # Comparación de optimizadores
│   └── train_rl_curriculum.py  # Auto-curriculum (Fase 5)
├── docs/                       # Documentación
│   └── proyecto.md             # Este archivo
├── pyproject.toml              # Configuración del proyecto
└── run_tests.py               # Script de tests
```

## 🧩 Fases Completadas

### ✅ Fase 1 - Microneuronas y Activaciones
- **Neuronas individuales** con pesos, bias y funciones de activación
- **Funciones de activación**: sigmoid, relu, tanh, linear
- **Derivadas** para backpropagation

### ✅ Fase 2 - Capas y Red Neuronal
- **Capas de neuronas** con conectividad completa
- **Red multicapa** con propagación forward/backward
- **Estructura modular** [inputs, hidden, ..., outputs]

### ✅ Fase 3 - Backpropagation y Trainer
- **Backpropagation completa** desde cero
- **Trainer** para entrenamiento supervisado
- **Validación de gradientes** y estabilidad

### ✅ Fase 4 - Optimizadores y Estabilidad
- **Optimizadores modulares**: SGD, Momentum, Adam, RMSprop
- **Tests de estabilidad** y validación de backprop
- **Comparación de optimizadores** en ejemplos

### ✅ Fase 5 - Variable Latente z y Proyector
- **Variable latente z** para planificación interna
- **LatentProjector** con proyección lineal + tanh
- **Integración completa** en NeuralNetwork
- **Ejemplos de entrenamiento** con variable latente

### ✅ Fase 6 - Mini Framework Autograd
- **Motor autograd completo** con propagación automática
- **Nodo Value** con sobrecarga de operadores
- **Operaciones matemáticas**: +, -, *, /, tanh, sigmoid, relu
- **API funcional**: linear, mse_loss
- **Entrenamiento XOR** usando autograd sin backprop manual
- **Comparación con backprop manual** validada

### ✅ Fase 7 - Macro-Neuronas Cognitivas
- **Celda de memoria diferenciable** con decaimiento temporal
- **Macro-neurona** que combina input + memoria + gating
- **Razonamiento secuencial** aprendido sin datasets
- **Sistema cognitivo** con contexto temporal
- **Demo de memoria** funcionando con autograd

## 🧠 Fase 7 - Macro-Neuronas Cognitivas (Sistema Razonador)

### 📁 Nueva Estructura:
```
src/
├── autograd/           # Motor diferenciable (Fase 6)
├── autograd/
│   ├── value.py          # Nodo escalar con autograd
│   ├── ops.py            # Funciones matemáticas
│   └── functional.py     # API estilo PyTorch
├── core/
│   └── adapters.py       # Enlace con red existente (futuro)
```

### 🔧 Componentes Implementados:

#### **Value - Nodo Base**
```python
from autograd.value import Value

# Crear valores con autograd
x = Value(2.0)
y = Value(3.0)
z = x * y + x.relu()  # Operaciones encadenadas
z.backward()  # Propagación automática
print(x.grad)  # Gradiente calculado automáticamente
```

#### **Operaciones Disponibles**
- **Aritméticas**: +, -, *, /, **
- **Activaciones**: tanh, sigmoid, relu, leaky_relu
- **Funciones**: exp, log
- **Red**: linear, mse_loss

#### **Ejemplo de Uso**
```python
# Entrenamiento XOR con autograd
from autograd.value import Value
from autograd.functional import linear, mse_loss

# Crear red con autograd
layer1 = make_layer(2, 4)  # Pesos como Value
layer2 = make_layer(4, 1)

# Forward automático
y_pred = forward(x)
loss = mse_loss(y_pred, y)
loss.backward()  # ¡Sin backprop manual!
```

### 🎯 Resultados Fase 6:
- **✅ Motor autograd funcional** sin dependencias
- **✅ Propagación automática** de gradientes
- **✅ API intuitiva** estilo PyTorch
- **✅ Entrenamiento XOR** convergente
- **✅ Validación** contra backprop manual

### 📊 Comparación de Enfoques:
| Enfoque | Backprop | Complejidad | Flexibilidad |
|---------|----------|-------------|--------------|
| Manual  | Manual   | Alta        | Baja         |
| Autograd| Automática| Baja        | Alta         |

### 🚀 Próximos Pasos:
- **Fase 7**: Macro-neuronas cognitivas
- **Fase 8**: Memoria y atención
- **Vectorización**: Optimización con NumPy (opcional)

## 🚀 Uso Rápido

### Ejemplo Básico - XOR
```python
from core.network import NeuralNetwork
from core import losses
from core.optimizers import Adam
from engine.trainer import Trainer

# Dataset XOR
dataset = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

# Crear red
nn = NeuralNetwork([2, 4, 1], activation="sigmoid")

# Entrenar
trainer = Trainer(
    nn,
    loss_fn=losses.mse_loss,
    loss_grad_fn=losses.mse_grad,
    optimizer=Adam(lr=0.01),
    batch_size=1
)

trainer.train(dataset, epochs=2000, verbose=True)
```

### Comparación de Optimizadores
```python
from core.optimizers import SGD, SGDMomentum, Adam, RMSprop

# Probar diferentes optimizadores
optimizers = [
    ("SGD", SGD(lr=0.1)),
    ("Momentum", SGDMomentum(lr=0.05, momentum=0.9)),
    ("Adam", Adam(lr=0.01)),
    ("RMSprop", RMSprop(lr=0.01)),
]

for name, optimizer in optimizers:
    trainer = Trainer(nn, losses.mse_loss, losses.mse_grad, optimizer=optimizer)
    trainer.train(dataset, epochs=1000)
```

## 🧪 Ejecución de Tests

### Tests de Validación
```bash
# Instalar dependencias
uv sync

# Ejecutar todos los tests
uv run python run_tests.py

# Tests individuales
uv run python -m pytest tests/test_stability.py -v
```

### Ejemplos
```bash
# XOR con Adam
uv run python examples/train_xor.py

# Comparar optimizadores
uv run python examples/compare_optimizers.py
```

## 📊 Arquitectura del Sistema

### Microneurona (Neuron)
```python
class Neuron:
    def __init__(self, n_inputs: int, activation: str = "sigmoid", optimizer: Optimizer = None):
        self.weights: List[float]  # Pesos sinápticos
        self.bias: float           # Sesgo
        self.activation: Activation # Función de activación
        self.optimizer: Optimizer   # Optimizador configurable
    
    def forward(self, inputs: List[float]) -> float:
        # Propagación hacia adelante
        pass
    
    def apply_gradients(self, dweights: List[float], dbias: float) -> None:
        # Actualización con optimizador
        pass
```

### Red Neuronal (NeuralNetwork)
```python
class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activation: str = "sigmoid"):
        # [n_inputs, n_hidden1, n_hidden2, ..., n_outputs]
        pass
    
    def forward(self, inputs: List[float], z: List[float] = None) -> List[float]:
        # Forward con soporte para variables latentes
        pass
    
    def train_step(self, inputs: List[float], targets: List[float], lr: float) -> float:
        # Un paso completo de entrenamiento
        pass
```

## 🎯 Características Implementadas

### ✅ Funcionalidad Base
- **Redes feedforward** multicapa
- **Backpropagation** completo
- **Funciones de activación** con derivadas
- **Funciones de pérdida** MSE y BCE
- **Optimización** con múltiples algoritmos

### ✅ Validación
- **Tests de estabilidad** con XOR
- **Verificación de convergencia**
- **Validación numérica**
- **Sin dependencias externas pesadas**

### ✅ Modularidad
- **Componentes intercambiables**
- **Optimizadores plug-and-play**
- **Funciones de activación extensibles**
- **Tests automatizados**

## 🚀 Próximos Pasos - Fase 5

### 🧠 Espacio Latente y Auto-Curriculum
- **Variables latentes z** para representación interna
- **Auto-curriculum learning** con RL
- **Generación de tareas** dinámica
- **Mente interna** para planificación

### 📈 Escalabilidad
- **Batch processing** con numpy
- **Paralelización** básica
- **Persistencia** de modelos
- **Visualización** de entrenamiento

## 📋 Requisitos

- **Python**: >= 3.12
- **Gestor de paquetes**: uv (recomendado)
- **Sistema operativo**: Linux/macOS/Windows

## 🏆 Logros

- ✅ **0 dependencias pesadas** - Python puro
- ✅ **100% modular** - Cada componente es intercambiable
- ✅ **Tests completos** - Validación exhaustiva
- ✅ **Documentación clara** - Código educativo
- ✅ **Ejemplos prácticos** - XOR y comparaciones

## 🎓 Propósito Educativo

Este proyecto sirve como:
- **Plataforma de aprendizaje** para redes neuronales
- **Base para experimentación** con arquitecturas
- **Demostración** de backpropagation desde cero
- **Puente** hacia frameworks más complejos

---

**Estado actual**: ✅ **Fase 4B Completada** - Sistema validado y listo para Fase 5
