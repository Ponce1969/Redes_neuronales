# 🧠 Fase 31-B: Cognitive Reasoning Layer

## 📋 Descripción

La **Cognitive Reasoning Layer** introduce un controlador inteligente que decide qué bloques del grafo cognitivo activar (y con qué intensidad) antes de ejecutar el grafo. Esto permite:

- **Rutas cognitivas selectivas**: El Reasoner decide dinámicamente qué caminos computacionales seguir
- **Eficiencia adaptativa**: Bloques irrelevantes pueden ser desactivados o atenuados
- **Razonamiento explícito**: Las decisiones del Reasoner son inspeccionables y visualizables

## 🏗️ Arquitectura

### Reasoner (MLP en NumPy)

```python
Input: z_plan por bloque (concatenados)
  ↓
Hidden Layer (tanh)
  ↓
Output Layer (n_blocks logits)
  ↓
Gating Strategy (softmax/topk/threshold)
  ↓
Gates aplicados a cada bloque
```

### Integración con CognitiveGraphHybrid

```python
# Forward normal
outputs = graph.forward(inputs)

# Forward con control selectivo
outputs = graph.forward_with_reasoner(inputs, reasoner, mode="softmax")
```

## 🚀 Uso Básico

### 1. Crear un Reasoner

```python
from core.reasoning import Reasoner

# n_inputs: tamaño concatenado de z_plan por bloque
# n_hidden: neuronas en capa oculta
# n_blocks: número de bloques en el grafo
reasoner = Reasoner(n_inputs=32, n_hidden=64, n_blocks=4, seed=42)
```

### 2. Ejecutar Inferencia Selectiva

```python
# Forward normal para computar planes latentes
_ = graph.forward({"sensor": [0.5, 0.5]})

# Forward con reasoner (decide gates)
outputs = graph.forward_with_reasoner(
    {"sensor": [0.5, 0.5]},
    reasoner,
    mode="softmax"  # o "topk", "threshold"
)

# Ver gates aplicados
print(graph.last_gates)
# {'sensor': 0.25, 'planner': 0.30, 'memory': 0.20, 'decision': 0.25}
```

### 3. Modos de Gating

#### Softmax (continuo)
```python
gates = reasoner.decide(z_list, mode="softmax", temp=1.0)
# Distribución suave: todos los bloques reciben algún peso
```

#### Top-K (sparse)
```python
gates = reasoner.decide(z_list, mode="topk", top_k=2)
# Solo los top-2 bloques se activan, resto = 0
```

#### Threshold (adaptativo)
```python
gates = reasoner.decide(z_list, mode="threshold")
# Solo bloques con logit > 0.1 se activan
```

## 🧬 Entrenamiento Evolutivo

### Estrategia Simple (1+λ)

```python
from core.reasoning import evolve_reasoner_on_task, evaluate_reasoner

# Dataset XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
Y = np.array([0, 1, 1, 0], dtype=np.float32)

# Evolucionar Reasoner
reasoner_evolved, loss_history = evolve_reasoner_on_task(
    graph=graph,
    base_reasoner=reasoner,
    X=X,
    Y=Y,
    generations=50,
    pop_size=10,
    mutation_scale=0.03,
    verbose=True
)

# Evaluar mejora
initial_loss = loss_history[0]
final_loss = loss_history[-1]
improvement = ((initial_loss - final_loss) / initial_loss) * 100
print(f"Mejora: {improvement:.2f}%")
```

### Funcionamiento

1. **Generación**: Crea `pop_size` mutantes del mejor Reasoner
2. **Evaluación**: Ejecuta cada mutante en el dataset y calcula MSE
3. **Selección**: Si algún mutante mejora, reemplaza al padre
4. **Repetir**: Por N generaciones

**Ventaja**: No requiere autograd, compatible con tu arquitectura actual.

## 📊 Análisis y Visualización

### Extraer Historial de Gates

```python
from core.reasoning import extract_gates_history

history = extract_gates_history(graph, reasoner, X, mode="softmax")

# history[i] = {'sensor': 0.25, 'planner': 0.30, ...} para X[i]
```

### Visualización en Dashboard

```python
# Los gates se guardan automáticamente en graph.last_gates
# Puedes colorear nodos en PyG/Plotly según gate:

import matplotlib.pyplot as plt
import networkx as nx

gates = graph.last_gates
colors = [gates[name] for name in graph.blocks.keys()]

nx.draw(G, node_color=colors, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Gate Activation')
```

## 🔧 Persistencia

### Guardar Reasoner Entrenado

```python
import numpy as np

# Guardar
state = reasoner.state_dict()
np.savez_compressed('reasoner_weights.npz', **state)

# Cargar
loaded = np.load('reasoner_weights.npz')
reasoner.load_state_dict(dict(loaded))
```

## 📈 Ejemplos

### Demo Básico
```bash
PYTHONPATH=src uv run python examples/cognitive_reasoning_demo.py
```

Muestra:
- Inferencia con diferentes modos de gating
- Comparación forward normal vs reasoner
- Gates aplicados por bloque

### Demo Evolutivo
```bash
PYTHONPATH=src uv run python examples/cognitive_reasoning_evolution_demo.py
```

Muestra:
- Entrenamiento evolutivo en XOR
- Curva de aprendizaje
- Análisis de gates aplicados
- Comparación predicciones antes/después

## 🧪 Tests

```bash
PYTHONPATH=src uv run pytest tests/test_reasoning.py -v
```

Cubre:
- Inicialización y configuración
- Modos de gating (softmax, topk, threshold)
- Mutación y serialización
- Integración con CognitiveGraphHybrid
- Entrenamiento evolutivo

## 🎯 Compatibilidad

El Reasoner es compatible con:

- ✅ **CognitiveBlock** (clásico con Value)
- ✅ **TRM_ACT_Block** (recursivo con Tensor)
- ✅ **LatentPlannerBlock** (con z_plan explícito)
- ✅ **ProjectionLayer** (AutoAlign)
- ✅ **AttentionRouter** (atención cognitiva)
- ✅ **CognitiveMonitor** (tracking de activaciones)

## 🔮 Próximos Pasos

### A Corto Plazo

1. **Integración con Dashboard**: Visualizar gates en tiempo real
   - Colorear nodos por gate en `dashboard_pyg_interactive.py`
   - Añadir heatmap temporal de gates en Streamlit

2. **Tracking en Monitor**: Registrar decisiones del Reasoner
   ```python
   monitor.track_gates(epoch, graph.last_gates)
   ```

3. **Multi-Objetivo**: Evolucionar considerando loss + eficiencia
   ```python
   fitness = loss + lambda * mean_gate  # Penalizar activación total
   ```

### A Largo Plazo

4. **Migración a PyTorch**: Reasoner diferenciable end-to-end
   ```python
   class TorchReasoner(nn.Module):
       # Backprop directo sobre gates
   ```

5. **Meta-Learning**: Reasoner aprende a aprender
   - Entrenar en múltiples tareas
   - Transfer learning entre grafos

6. **Reasoner Jerárquico**: Control multi-nivel
   - Meta-Reasoner decide qué sub-grafos activar
   - Sub-Reasoners controlan bloques individuales

## 📚 Archivos Implementados

```
src/core/reasoning/
├── __init__.py              # Exportaciones del módulo
├── reasoner.py              # Clase Reasoner (MLP + mutación)
└── training.py              # Utilidades de entrenamiento evolutivo

examples/
├── cognitive_reasoning_demo.py           # Demo básico
└── cognitive_reasoning_evolution_demo.py # Demo con entrenamiento

tests/
└── test_reasoning.py        # Suite completa de tests

docs/
└── fase31b_reasoning_layer.md  # Esta documentación
```

## 💡 Tips de Uso

### 1. Dimensionamiento del Reasoner

```python
# Rule of thumb:
n_inputs = max_plan_dim * n_blocks  # Con padding automático
n_hidden = 2 * n_inputs            # Capacidad expresiva
```

### 2. Escala de Mutación

```python
# Exploración agresiva: scale=0.05
# Refinamiento fino: scale=0.01
# Balance recomendado: scale=0.03
```

### 3. Población Evolutiva

```python
# Pocos bloques (3-5): pop_size=8
# Grafos medianos (6-10): pop_size=12
# Grafos grandes (>10): pop_size=16
```

### 4. Debugging

```python
# Verificar gates suman ~1.0 en softmax
gates = reasoner.decide(z_list, mode="softmax")
print(f"Sum: {sum(gates.values())}")  # Debe ser ≈ 1.0

# Verificar activación real de bloques
for name, block in graph.blocks.items():
    print(f"{name}: act={block.last_activation:.4f}, gate={graph.last_gates[name]:.4f}")
```

## 🎓 Conceptos Clave

### Gate vs Activation

- **Gate**: Peso decidido por el Reasoner (antes del forward)
- **Activation**: Salida del bloque (después del forward)
- **Relación**: `activation_effective = activation_raw * gate`

### Top-K vs Threshold

- **Top-K**: Garantiza exactamente K bloques activos (sparse determinista)
- **Threshold**: Número variable según confianza (sparse adaptativo)
- **Softmax**: Todos activos con pesos variables (denso)

### Evolución vs Gradientes

- **Ventaja evolución**: No requiere diferenciabilidad, explora bien
- **Ventaja gradientes**: Más eficiente en grandes dimensiones
- **Recomendación**: Usa evolución ahora, migra a PyTorch si necesitas escalar

---

**Implementación completada**: Fase 31-B ✅  
**Autor**: Neural Core Project  
**Fecha**: Noviembre 2024  
**Versión**: 1.0
