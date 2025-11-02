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
│   ├── __init__.py                 # Alias automáticos (autograd/core/engine)
│   ├── autograd/
│   │   ├── __init__.py             # Reexporta Value
│   │   ├── value.py                # Nodo autograd
│   │   ├── functional.py           # linear, mse_loss, etc.
│   │   └── ops.py                  # Operaciones auxiliares
│   ├── core/autograd_numpy/
│   │   ├── __init__.py             # Tensor + pérdidas vectorizadas (Fase 10)
│   │   ├── tensor.py               # Motor NumPy minimalista
│   │   └── loss.py                 # MSE / BCE vectorizados
│   ├── core/
│   │   ├── __init__.py             # Componentes cognitivos
│   │   ├── memory_cell.py          # Celda de memoria diferenciable
│   │   ├── macro_neuron.py         # Macro-neurona con gating
│   │   ├── reasoning_unit.py       # Unidad de razonamiento
│   │   ├── cognitive_block.py      # Bloque cognitivo modular
│   │   ├── cognitive_graph.py      # Grafo de bloques cognitivos
│   │   ├── trm_block.py            # Tiny Recursive Model (Fase 10)
│   │   ├── trm_act_block.py        # TRM con ACT + deep supervision (Fase 11)
│   │   ├── cognitive_graph_trm.py  # Grafo TRM adaptativo (Fase 12)
│   │   ├── cognitive_graph_hybrid.py # Grafo híbrido (Fase 13)
│   │   └── projection_layer.py     # AutoAlign (Fase 14)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Entrenamiento supervisado
│   │   ├── rl_trainer.py           # Entrenamiento RL
│   │   ├── dataset.py              # Utilidades de datasets
│   │   └── predictor.py            # Predictores utilitarios
│   └── app.py                      # Punto de entrada CLI
├── examples/
│   ├── cognitive_agent_demo.py     # Bloque cognitivo secuencial
│   ├── cognitive_graph_demo.py     # Grafo cognitivo (Fase 9)
│   ├── trm_demo.py                 # XOR con TRM vectorizado (Fase 10)
│   ├── trm_act_demo.py             # TRM con halting adaptativo (Fase 11)
│   ├── trm_cognitive_graph_demo.py # Grafo TRM recursivo (Fase 12)
│   ├── hybrid_graph_demo.py        # Grafo híbrido TRM + CognitiveBlock (Fase 13)
│   └── hybrid_graph_autoalign_demo.py # AutoAlign dinámico (Fase 14)
├── tests/
│   ├── test_network.py
│   ├── test_neuron.py
│   ├── test_trainer.py
│   └── test_cognitive_graph_hybrid.py
├── docs/
│   └── proyecto.md                 # Documentación general
├── pyproject.toml                  # Configuración del proyecto
└── README.md
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

### ✅ Fase 8 - CognitiveBlock (Arquitectura Cognitiva Modular)
- **ReasoningUnit**: Unidad que combina percepción y memoria para inferencias
- **CognitiveBlock**: Bloque cognitivo completo con percepción, memoria y razonamiento
- **Arquitectura modular**: Componentes interconectables para construir mentes artificiales
- **Demo de predicción secuencial**: Aprende patrones temporales sin supervisión
- **Integración completa**: Todos los componentes usan el motor autograd

### ✅ Fase 9 - CognitiveGraph (Mente Modular Emergente)
- **CognitiveGraph**: Red de CognitiveBlock interconectados
- **Comunicación interbloques**: Feedforward, recurrente y reflexivo
- **Memoria compartida**: Estado global accesible por todos los bloques
- **Demo `cognitive_graph_demo.py`**: Mente artificial con percepción → razonamiento → decisión
- **Semilla determinista**: `random.seed(42)` para reproducibilidad
- **Alias automáticos**: `src/__init__.py` expone `autograd`, `core` y `engine`

### ✅ Fase 10 - Motor Tensor Vectorizado + TRM Base
- **Tensor** NumPy (`core/autograd_numpy`) como reemplazo de `Value` para operaciones vectorizadas
- **Funciones de pérdida** MSE/BCE adaptadas al nuevo motor
- **TRMBlock** recursivo con estado latente z y detach entre pasos
- **Demo `examples/trm_demo.py`**: TRM aprende XOR con actualización aproximada

### ✅ Fase 11 - Deep Supervision + Adaptive Computation Time
- **TRM_ACT_Block** con neurona de halting y cálculo adaptativo de pasos
- **Deep supervision** en cada iteración con pérdidas parciales
- **Demo `examples/trm_act_demo.py`** validando razonamiento adaptativo en XOR

### ✅ Fase 12 - CognitiveGraph TRM
- **CognitiveGraphTRM** para orquestar múltiples TRM_ACT conectados
- **Step numérico** y reset de estados para simulaciones recursivas
- **Demo `examples/trm_cognitive_graph_demo.py`**: pipeline percepción → razonamiento → decisión
- **Tests `tests/test_trm_cognitive_graph.py`** asegurando estabilidad y resets

### ✅ Fase 13 - CognitiveGraph Hybrid
- **CognitiveGraphHybrid** integra CognitiveBlock clásicos y TRM_ACT adaptativos
- **Compatibilidad bidireccional**: convierte salidas entre Value ↔ Tensor automáticamente
- **Demo `examples/hybrid_graph_demo.py`** demostrando razonamiento mixto
- **Tests `tests/test_cognitive_graph_hybrid.py`** validando reset y estabilidad

### ✅ Fase 14 - AutoAlign Layers
- **ProjectionLayer** genera proyecciones lineales aprendibles entre bloques con dimensiones distintas
- **CognitiveGraphHybrid** ahora crea proyecciones on-the-fly al conectar nodos con tamaños incompatibles
- **Demo `examples/hybrid_graph_autoalign_demo.py`** muestra conexiones sensor → memory → reasoner → decision con AutoAlign
- **Tests `tests/test_cognitive_graph_hybrid.py`** cubren estabilidad con AutoAlign activado

## 🧠 Estructura Completa del Proyecto

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
- **Fase 10**: Sistemas cognitivos multi-agente
- **Vectorización**: Optimización con NumPy (opcional)
- **Persistencia**: Guardado/carga de pesos
- **Exportar más alias**: Evaluar exposición plana de `io`, `examples`

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

## 🚀 Próximos Pasos - Fase 15

### 🧠 Entrenamiento Conjunto AutoAlign + CognitiveGraph
- **Optimización compartida**: combinar gradientes de Value y Tensor en presencia de proyecciones
- **Persistencia** de pesos de ProjectionLayer y estados vectorizados
- **Interfaz común** para mezclar bloques clásicos, TRM, memoria y nuevas capas

### 📈 Escalabilidad
- **Batch processing** con NumPy para TRM y grafo cognitivo
- **Estadísticas de halting** y visualización de pasos de razonamiento
- **Persistencia** de modelos y replay de grafos cognitivos

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

**Estado actual**: ✅ **Fase 14 Completada** - AutoAlign funcionando en CognitiveGraphHybrid
