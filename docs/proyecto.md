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
│   │   ├── projection_layer.py     # AutoAlign (Fase 14)
│   │   ├── training/               # Entrenamiento global (Fase 15)
│   │   │   ├── __init__.py         # Alias utilitarios de entrenamiento
│   │   │   ├── losses.py           # MSE, L1, BCE vectorizados
│   │   │   ├── optimizers.py       # SGD / Adam híbridos Value-Tensor
│   │   │   └── trainer.py          # GraphTrainer con deep supervision
│   │   ├── attention/              # Atención cognitiva dinámica (Fase 16)
│   │   │   ├── __init__.py         # Exportaciones de atención
│   │   │   ├── attention_layer.py  # Capa de atención Query-Key-Value
│   │   │   ├── attention_router.py # Router de múltiples atenciones
│   │   │   └── utils.py            # Softmax y utilidades numéricas
│   │   └── monitor/                # Cognitive Monitor System (Fase 17)
│   │       ├── __init__.py         # Exportaciones de monitoreo
│   │       ├── cognitive_monitor.py# Seguimiento de activaciones/atención
│   │       ├── logger.py           # Logger JSON/timestamps
│   │       └── visualizer_streamlit.py # Dashboard interactivo (Fase 19)
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
│   ├── hybrid_graph_autoalign_demo.py # AutoAlign dinámico (Fase 14)
│   ├── global_training_demo.py     # Entrenamiento global con deep supervision (Fase 15)
│   └── cognitive_attention_demo.py # Atención cognitiva dinámica (Fase 16)
├── dashboard/
│   └── app_dashboard.py            # App Streamlit del Cognitive Dashboard (Fase 19)
├── tests/
│   ├── test_network.py
│   ├── test_neuron.py
│   ├── test_trainer.py
│   ├── test_cognitive_graph_hybrid.py
│   ├── test_graph_trainer.py
│   └── test_attention_router.py    # (Fase 16-17 - futuro)
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

### ✅ Fase 15 - Deep Supervision Training Loop
- **GraphTrainer** entrena CognitiveGraphHybrid completo con supervisión profunda
- **Módulos `core/training`** centralizan pérdidas, optimizadores y entrenamiento global
- **Compatibilidad híbrida**: actualiza simultáneamente CognitiveBlock, TRM_ACT_Block y ProjectionLayer
- **Demo `examples/global_training_demo.py`** aprende XOR extremo a extremo
- **Tests `tests/test_graph_trainer.py`** validan recolección de parámetros y paso de entrenamiento

### ✅ Fase 16 - Cognitive Attention System (CAS)
- **CognitiveAttentionLayer** calcula atención contextual Query-Key-Value entre bloques
- **AttentionRouter** coordina pesos dinámicos para cada conexión del grafo
- **CognitiveGraphHybrid** integra atención + AutoAlign para foco cognitivo adaptativo
- **Demo `examples/cognitive_attention_demo.py`** muestra cómo varía el foco en tiempo real
- **Tests `tests/test_cognitive_graph_hybrid.py`** verifican almacenamiento y normalización de pesos de atención

### ✅ Fase 17 - Cognitive Monitor System (CMS)
- **CognitiveMonitor** registra activaciones, pesos de atención y pérdidas en tiempo real
- **CognitiveLogger** provee logging estructurado en consola/JSON con timestamps
- **Integración** desde CognitiveGraphHybrid y GraphTrainer para telemetría continua
- **Demo `examples/cognitive_monitor_demo.py`** ejecuta entrenamiento XOR monitorizado
- **Datos persistentes** listos para dashboards (Streamlit en Fase 19)

### ✅ Fase 18 - Memory Replay System (MRS)
- **EpisodicMemory** almacena inputs, targets, outputs, pérdidas y mapas de atención
- **MemoryReplaySystem** consolida experiencias exitosas mediante sleep cycles
- **GraphTrainer** registra cada episodio automáticamente durante el entrenamiento
- **Fase de sueño** con `sleep_and_replay()` que reduce la pérdida promedio
- **Demo `examples/memory_replay_demo.py`** muestra consolidación tras 300 épocas

### ✅ Fase 19 - Cognitive Dashboard (Streamlit)
- **CognitiveVisualizer** renderiza pérdidas, activaciones, atención y memoria en tiempo real
- **App Streamlit** `dashboard/app_dashboard.py` consume el grafo activo vía `st.session_state`
- **Integración opcional**: demos pueden lanzar el dashboard en segundo plano
- **Dependencias añadidas**: `streamlit`, `pandas`, `plotly`, `altair`, `pydeck`
- **Interfaz** multipestaña con métricas clave actualizadas durante entrenamiento y sleep cycles

### ✅ Fase 20 - Meta-Learning Loop
- **Paquete `core.meta`** con reglas adaptativas (`adaptive_lr`, `adaptive_focus`, `adaptive_sleep`) que observan pérdidas y atenciones del monitor
- **MetaLearningController** ajusta dinámicamente learning rate, foco atencional e intervalo de consolidación usando el monitor y el MemoryReplaySystem
- **Demo `examples/meta_learning_demo.py`** (`PYTHONPATH=src uv run python examples/meta_learning_demo.py`) muestra el bucle autorregulado en acción
- **Tests `tests/test_meta_rules.py`** validan las heurísticas de ajuste
- Detalles ampliados en `docs/fase20_meta_loop.md`

### ✅ Fase 21 - Cognitive Evolution System (CES)
- **Paquete `core.evolution`** con `CognitivePopulation`, utilidades de crossover y el `EvolutionManager` para coordinar generaciones
- **Crossover híbrido** que mezcla pesos tipo Value/Tensor con mutaciones ligeras para mantener diversidad
- **Evolución generacional**: selección de los grafos con mejor fitness, cruce y regeneración automática de la población
- **Demo `examples/cognitive_evolution_demo.py`** (`PYTHONPATH=src uv run python examples/cognitive_evolution_demo.py`) ejecuta varias generaciones sobre XOR
- Permite experimentar con evolución de arquitecturas sin alterar el flujo de entrenamiento base

### ✅ Fase 22 - Cognitive Society System (CSS)
- **Paquete `core.society`** con `CognitiveAgent`, `CommunicationChannel` y `SocietyManager` para coordinar agentes múltiples
- **Intercambio social**: agentes comparten experiencias vía `exchange_memories` y broadcasts de mejores episodios
- **Cooperación adaptativa**: cada agente entrena su propio grafo pero se beneficia de memorias ajenas
- **Demo `examples/cognitive_society_demo.py`** (`PYTHONPATH=src uv run python examples/cognitive_society_demo.py`) muestra cómo convergen las pérdidas compartiendo conocimiento

### ✅ Fase 23A - Cognitive API Server (CAS)
- **Paquete `api/`** con servidor FastAPI, routers modulares y estado compartido (`CognitiveAppState`)
- **Endpoints REST**: `/predict`, `/feedback`, `/evolve`, `/status` para interactuar con la sociedad en tiempo real
- **Integración con `.env`**: API key y parámetros de despliegue para una configuración segura en la Orange Pi
- **Script de arranque** (via `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`), listo para exponerse tras un reverse proxy HTTPS
- **Monitoreo en vivo**: `/status` reporta pérdidas recientes, memoria y espectro de agentes activos

### ✅ Fase 23B - Cognitive Persistence Layer (CPL)
- **Paquete `core.persistence`** con gestores de rutas, serialización de pesos/memorias y `PersistenceManager`
- **Formato ligero**: pesos en `.npz` comprimido, memorias y métricas en `.json` human-readable
- **Integración con API**: carga automática al iniciar servidor y endpoint `/save` para persistencia manual o vía cron
- **Directorios `data/persistence/weights|memories`** almacenan hasta 100 episodios recientes por agente
- Asegura continuidad del aprendizaje tras reinicios o despliegues en nodos distribuidos

### ✅ Fase 23C - Cognitive Network (Distribución de Agentes)
- **Paquete `core.distribution`** con `CognitiveDistributor` (cliente HTTP) y helpers de recepción
- **Endpoint `/share`** en FastAPI para sincronizar memorias/pesos entre nodos protegidos por API key
- **Serialización remota**: transferencias usan `.npz` base64 y memorias recientes en JSON (límite configurable)
- **Tests `tests/test_distribution.py`** validan generación de payload y aplicación remota en entornos aislados
- Permite interconectar Orange Pi, servidores cloud o PCs formando una red de sociedades cognitivas cooperativas

### ✅ Fase 24 - Cognitive Federation (Aprendizaje Federado)
- **Paquete `core.federation`** con utilidades de serialización/promedio y `FederatedClient`
- **Router `/federate`** en FastAPI (servidor cloud) agrega pesos (`/upload`) y entrega promedio global (`/global`)
- **Seguridad**: dependencia `require_api_key` y helper `get_api_headers` reutilizados por distribuidores y clientes
- **Tests `tests/test_federation.py`** cubren serialización, promedio y roundtrip cliente-servidor (requiere FastAPI instalado)
- Permite que nodos locales entrenen con sus datos y sincronicen pesos con un nodo federador sin exponer datos crudos

### ✅ Fase 25 - Cognitive Scheduler (Ciclo Autónomo)
- **Paquete `core.scheduler`** con `SchedulerConfig` (intervalos/flags) y `CognitiveScheduler` en hilo daemon
- **Ciclos automatizados**: entrenamiento, persistencia, federación opcional, intercambio de memorias y sueño cognitivo
- **Integrado en `api/dependencies.py`**: se instancia en el arranque, reutilizando `PersistenceManager` y `FederatedClient`
- **Configuración flexible** (`loop_sleep`, banderas `enable_*`) permite desactivar federación/evolución en nodos aislados
- Diseñado para mantener la sociedad aprendiendo sin intervención manual, alineado con despliegues Orange Pi + nube

#### ▶️ Cómo lanzar el dashboard

1. Inicia el proceso combinado desde la raíz del proyecto:
   ```bash
   PYTHONPATH=src uv run python launch_cognitive.py
   ```
   Este script ejecuta el entrenamiento (demo `memory_replay_demo.py`) y levanta Streamlit en `http://localhost:8501`, persistiendo los snapshots en `dashboard_state.json`.

2. Abre el navegador en `http://localhost:8501` para visualizar las pestañas de **Pérdidas**, **Activaciones**, **Atención** y **Memoria episódica**. El dashboard consumirá datos en vivo si el entrenamiento sigue corriendo o mostrará el último snapshot disponible.

También puedes ejecutar los pasos manualmente si prefieres procesos separados:
```bash
# Terminal 1 – entrenamiento
PYTHONPATH=src uv run python examples/memory_replay_demo.py

# Terminal 2 – dashboard
PYTHONPATH=src uv run streamlit run dashboard/app_dashboard.py
``` 
Ambas variantes leen/escriben el snapshot compartido (`dashboard_state.json`), por lo que la visualización se mantiene incluso cuando el entrenamiento se detiene.




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

## 🚀 Próximos Pasos - Fase 20

### 🧠 Memory Replay Dashboard Avanzado
- **Controles interactivos** para filtrar episodios y ajustar replay_factor
- **Comparativa de sesiones** con descargas CSV desde Streamlit

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


