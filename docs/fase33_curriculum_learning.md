```markdown
# 📚 Fase 33 - Curriculum Learning System

## 🎯 Objetivo

Implementar un sistema completo de **Curriculum Learning** que permita al Reasoner aprender de manera progresiva, desde tareas simples hasta complejas, imitando el proceso de aprendizaje humano.

---

## 🧠 Concepto Base

El **Curriculum Learning** divide el aprendizaje en etapas graduadas de dificultad. Cada etapa:
- Tiene su propio dataset o función objetivo
- Evalúa el Reasoner con métricas avanzadas
- Evoluciona el Reasoner mediante mutación y selección
- Cuando alcanza un rendimiento mínimo → pasa al siguiente nivel

```
Progresión típica:
identity → xor → parity(3) → counting → sequence → memory → reasoning
   (1)      (2)      (3)        (4)        (5)       (6)       (7)
```

---

## 📁 Estructura de Archivos

```
src/core/curriculum/
├── __init__.py                    # Exportaciones del módulo
├── tasks.py                       # Generadores de tareas ⭐
├── metrics.py                     # Métricas cognitivas avanzadas ⭐
├── curriculum_stage.py            # Definición de etapas ⭐
├── curriculum_manager.py          # Manager principal ⭐
├── evaluator.py                   # Evaluador integrado con grafo ⭐
└── checkpointer.py                # Sistema de checkpoints ⭐

src/api/routes/
└── curriculum.py                  # API REST para curriculum ⭐

dashboard/
└── dashboard_curriculum.py        # Dashboard Streamlit ⭐

examples/
└── curriculum_learning_demo.py    # Demo standalone ⭐

tests/
└── test_curriculum.py             # Tests unitarios ⭐

docs/
└── fase33_curriculum_learning.md  # Esta documentación
```

---

## 🧩 Componentes Principales

### 1️⃣ **Tasks (`tasks.py`)**

Generadores de tareas de dificultad creciente.

#### Tareas Disponibles:

| Tarea | Dificultad | Descripción | Input → Output |
|-------|------------|-------------|----------------|
| **identity** | 1/10 | Copiar entrada sin transformación | `(n,) → (n,)` |
| **xor** | 2/10 | XOR clásico (no lineal básico) | `(2,) → (1,)` |
| **parity** | 3/10 | Paridad de N bits | `(n,) → (1,)` |
| **counting** | 4/10 | Contar 1s en entrada binaria | `(n,) → (1,)` |
| **sequence** | 5/10 | Predecir siguiente elemento | `(n,) → (n,)` |
| **memory** | 6/10 | Recordar primer elemento | `(n,) → (1,)` |
| **reasoning** | 7/10 | Lógica compuesta AND/OR/NOT | `(3,) → (1,)` |

#### Ejemplo de Uso:

```python
from core.curriculum import tasks

# Generar dataset XOR
X, Y = tasks.xor_task(samples=16)
# X.shape = (16, 2), Y.shape = (16, 1)

# O usar el registro
task_func = tasks.get_task("parity")
X, Y = task_func(n_bits=3, samples=32)
```

---

### 2️⃣ **Metrics (`metrics.py`)**

Métricas avanzadas para evaluación cognitiva.

#### Métricas Implementadas:

1. **MSE Loss**: Error cuadrático medio (principal)
2. **MAE Loss**: Error absoluto medio
3. **Accuracy**: Precisión en clasificación binaria
4. **Gate Diversity**: Uniformidad en el uso de bloques
5. **Gate Entropy**: Entropía de Shannon de los gates
6. **Gate Utilization**: % de bloques activos (>0.1)
7. **Convergence Rate**: Velocidad de mejora (primeros vs últimos)
8. **Stability**: Inverso de la varianza del error

#### Ejemplo de Uso:

```python
from core.curriculum import CognitiveMetrics

predictions = np.array([...])
targets = np.array([...])
gates_history = [np.array([0.5, 0.3, 0.2]), ...]

metrics = CognitiveMetrics.compute_all(
    predictions,
    targets,
    gates_history
)

print(metrics)
# {
#   'mse_loss': 0.0234,
#   'accuracy': 0.875,
#   'gate_diversity': 0.823,
#   'gate_entropy': 1.045,
#   'convergence_rate': 0.012,
#   'stability': 0.943
# }
```

---

### 3️⃣ **CurriculumStage (`curriculum_stage.py`)**

Define una etapa individual del curriculum.

#### Parámetros:

- `name`: Nombre descriptivo
- `task_generator`: Función que genera (X, Y)
- `difficulty`: Nivel 1-10
- `max_epochs`: Máximo de épocas
- `success_threshold`: Loss para considerar completada
- `fail_threshold`: Loss para no fallar totalmente
- `log_interval`: Cada cuántas épocas loggear
- `evolution_generations`: Generaciones de evolución por época
- `mutation_scale`: Escala de mutación

#### Ejemplo de Uso:

```python
from core.curriculum import CurriculumStage, tasks

stage = CurriculumStage(
    name="xor",
    task_generator=lambda: tasks.xor_task(samples=16),
    difficulty=2,
    max_epochs=50,
    success_threshold=0.02,
    fail_threshold=0.15,
    log_interval=10,
)
```

#### Curriculum Estándar:

```python
from core.curriculum import create_standard_curriculum

stages = create_standard_curriculum()
# Retorna lista de 7 etapas pre-configuradas
```

---

### 4️⃣ **CurriculumEvaluator (`evaluator.py`)**

Evalúa el Reasoner en tareas específicas.

#### Características:

- ✅ Integración con `CognitiveGraphHybrid`
- ✅ Tracking de historial de gates
- ✅ Métricas automáticas avanzadas
- ✅ Manejo robusto de errores

#### Ejemplo de Uso:

```python
from core.curriculum import CurriculumEvaluator

evaluator = CurriculumEvaluator(graph, mode="softmax")

metrics = evaluator.evaluate(
    reasoner=reasoner_manager.reasoner,
    X=X_train,
    Y=Y_train,
    track_gates=True
)

print(metrics['mse_loss'])  # 0.0234
```

---

### 5️⃣ **CurriculumManager (`curriculum_manager.py`)**

Manager principal que coordina todo el entrenamiento.

#### Características Profesionales:

- ✅ **Sin variables globales** (usa inyección de dependencias)
- ✅ **Checkpointing automático** después de cada etapa
- ✅ **Resume** desde última etapa completada
- ✅ **Thread-safe** con RLock
- ✅ **Early stopping** inteligente
- ✅ **Métricas en tiempo real**
- ✅ **Logging estructurado**

#### Ejemplo de Uso:

```python
from core.curriculum import CurriculumManager, CurriculumStage, tasks

# Crear manager (inyección de dependencias)
manager = CurriculumManager(
    reasoner_manager=reasoner_manager,
    graph=graph,
    auto_save=True
)

# Añadir etapas
manager.add_stage(CurriculumStage("identity", tasks.identity_task, 1))
manager.add_stage(CurriculumStage("xor", tasks.xor_task, 2))

# Ejecutar
history = manager.run()

# Ver resultados
for record in history:
    print(f"{record['stage']}: loss={record['mse_loss']:.4f}")
```

---

### 6️⃣ **CurriculumCheckpointer (`checkpointer.py`)**

Sistema de checkpoints automáticos.

#### Características:

- ✅ Auto-save después de cada etapa
- ✅ Resume desde última etapa
- ✅ Versionado de checkpoints
- ✅ Backup automático (mantiene últimos 5)
- ✅ Guarda estado del Reasoner + metadata

#### Ubicación:

```
data/curriculum/
├── curriculum_state.json           # Estado actual
└── backups/
    ├── curriculum_state_20250106_223045.json
    ├── curriculum_state_20250106_223012.json
    └── ...
```

---

## 🌐 API REST

### Endpoints Disponibles:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/curriculum/start` | Inicia el curriculum |
| GET | `/curriculum/status` | Obtiene estado actual |
| POST | `/curriculum/pause` | Pausa ejecución |
| POST | `/curriculum/resume` | Reanuda ejecución |
| POST | `/curriculum/reset` | Resetea a etapa 0 |
| GET | `/curriculum/history` | Obtiene historial completo |
| GET | `/curriculum/checkpoints` | Lista checkpoints disponibles |
| POST | `/curriculum/export` | Exporta resultados |

### Ejemplos de Uso:

#### Iniciar Curriculum Estándar:

```bash
curl -X POST http://localhost:8000/curriculum/start \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Iniciar Curriculum Personalizado:

```bash
curl -X POST http://localhost:8000/curriculum/start \
  -H "Content-Type: application/json" \
  -d '{
    "stages": [
      {
        "name": "identity",
        "difficulty": 1,
        "max_epochs": 30,
        "success_threshold": 0.02,
        "fail_threshold": 0.15
      },
      {
        "name": "xor",
        "difficulty": 2,
        "max_epochs": 50,
        "success_threshold": 0.03,
        "fail_threshold": 0.2
      }
    ]
  }'
```

#### Monitorear Progreso:

```bash
# Loop de monitoreo
watch -n 2 'curl -s http://localhost:8000/curriculum/status | jq ".progress, .current_stage_name"'
```

#### Ver Historial:

```bash
curl http://localhost:8000/curriculum/history | jq
```

---

## 🎨 Dashboard Streamlit

### Lanzar Dashboard:

```bash
PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py
```

### Características:

1. **Estado General**:
   - Progress bar en tiempo real
   - Métricas principales (estado, etapa actual, completadas)
   - Indicadores visuales (🟢 running, 🔴 stopped, ⏸️  paused)

2. **Visualizaciones**:
   - Gráfico de evolución del loss
   - Bar chart de epochs por etapa
   - Line chart de accuracy (si disponible)

3. **Tabla Detallada**:
   - Status, nombre, dificultad, epochs, loss, accuracy
   - Indicador de completitud (✅ completo, ⚠️  parcial)

4. **Controles**:
   - ▶️ Start, ⏸️  Pause, 🔄 Reset
   - Auto-refresh configurable (1-10s)
   - Selección de presets (Estándar, Rápido, Avanzado)

5. **Estadísticas Globales**:
   - Total epochs
   - Average loss
   - Average accuracy
   - Completion rate

---

## 🚀 Guía de Uso

### Opción 1: Demo Standalone

```bash
# Ejecutar demo local (sin servidor)
PYTHONPATH=src python examples/curriculum_learning_demo.py
```

### Opción 2: Con Servidor + Dashboard

```bash
# Terminal 1: Servidor FastAPI
PYTHONPATH=src uv run uvicorn api.server:app --reload

# Terminal 2: Dashboard
PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py

# Terminal 3: Iniciar curriculum via API
curl -X POST http://localhost:8000/curriculum/start
```

### Opción 3: Integrado en Código

```python
from core.curriculum import CurriculumManager, create_standard_curriculum
from core.reasoning.reasoner_manager import ReasonerManager
from core.cognitive_graph_hybrid import CognitiveGraphHybrid

# Setup
graph = CognitiveGraphHybrid()
reasoner_mgr = ReasonerManager(n_inputs=24, n_hidden=48, n_blocks=3)

# Curriculum
manager = CurriculumManager(reasoner_mgr, graph)

for stage in create_standard_curriculum():
    manager.add_stage(stage)

# Run
history = manager.run()

# Analyze
for record in history:
    print(f"{record['stage']:15s} | loss={record['mse_loss']:.4f} | epochs={record['epochs']}")
```

---

## 📊 Flujo de Trabajo Típico

```
1. Servidor arranca
   └─> ReasonerManager + Graph inicializados en CognitiveAppState

2. Usuario abre Dashboard
   └─> Selecciona preset (Estándar / Rápido / Avanzado / Personalizado)

3. Click en "▶️ Start"
   └─> POST /curriculum/start
       └─> CurriculumManager crea etapas
           └─> Lanza en background

4. Loop de entrenamiento (cada etapa):
   ├─> Genera dataset con task_generator()
   ├─> Evalúa Reasoner actual
   ├─> Evoluciona con mutación ligera
   ├─> Log cada N epochs
   ├─> Early stopping si alcanza threshold
   └─> Auto-save checkpoint

5. Dashboard auto-refresh:
   ├─> GET /curriculum/status cada 2s
   └─> Actualiza gráficos y métricas

6. Completion:
   ├─> Todas las etapas completadas
   ├─> Reasoner final guardado
   └─> Historial disponible en /curriculum/history
```

---

## 🎯 Comparación con Fase 32

| Aspecto | Fase 32 (Reasoner Integration) | Fase 33 (Curriculum Learning) |
|---------|-------------------------------|-------------------------------|
| **Objetivo** | Integrar Reasoner con API/Dashboard | Entrenar Reasoner progresivamente |
| **Aprendizaje** | Evolución en task único (XOR) | Evolución en múltiples tasks secuenciales |
| **Métricas** | MSE Loss básico | 8 métricas cognitivas avanzadas |
| **Checkpointing** | Manual (save/load) | Automático después de cada etapa |
| **Generalización** | Especializado en una tarea | Generaliza a través de curriculum |
| **Observabilidad** | Estado y gates | + Progreso, historial, convergencia |

---

## 🧪 Testing

### Ejecutar Tests:

```bash
# Todos los tests de curriculum
pytest tests/test_curriculum.py -v

# Test específico
pytest tests/test_curriculum.py::test_xor_task -v

# Con coverage
pytest tests/test_curriculum.py --cov=src/core/curriculum --cov-report=html
```

### Tests Implementados:

- ✅ Generación correcta de todas las tareas
- ✅ Cálculo preciso de métricas
- ✅ Validación de parámetros de CurriculumStage
- ✅ Creación y reset de CurriculumManager
- ✅ Evaluación con CurriculumEvaluator
- ✅ Test de integración end-to-end

---

## 🔥 Beneficios del Sistema

### 1️⃣ **Aprendizaje Progresivo**
- El Reasoner aprende gradualmente, como un humano
- Evita overfitting en tareas simples
- Mejora generalización

### 2️⃣ **Observabilidad Total**
- Dashboard en tiempo real
- Métricas avanzadas (no solo loss)
- Historial completo de progreso

### 3️⃣ **Robustez**
- Checkpointing automático
- Resume desde última etapa
- Manejo de fallos graceful

### 4️⃣ **Flexibilidad**
- Curriculum personalizable
- Tareas extensibles
- Thresholds ajustables

### 5️⃣ **Integración Limpia**
- Sin variables globales
- Inyección de dependencias
- API REST completa

---

## 📚 Próximos Pasos Sugeridos

Después de dominar la Fase 33, puedes:

### Fase 34: **Benchmark Suite**
- Suite de benchmarks reproducibles
- Comparar diferentes configuraciones de Reasoner
- Reportes automáticos con gráficos

### Fase 35: **Federated Reasoners**
- Conectar múltiples nodos (Orange Pi + Cloud)
- Sincronización de experiencias
- Aprendizaje colaborativo distribuido

### Fase 36: **Meta-Learning**
- Reasoner que aprende a aprender
- Transfer learning entre tareas
- Few-shot adaptation

---

## 🎓 Lecciones Aprendidas

### ✅ **Lo que Funciona Bien**:
1. Curriculum estándar con 7 etapas
2. Early stopping con success_threshold
3. Evolución ligera (1-2 generaciones por época)
4. Auto-save después de cada etapa
5. Métricas avanzadas para diagnóstico

### ⚠️ **Desafíos Comunes**:
1. **Thresholds muy bajos**: Nunca alcanza success, se detiene en fail
2. **Mutation scale grande**: Inestabilidad, divergencia
3. **Epochs insuficientes**: No aprende, pasa parcialmente
4. **Tasks incompatibles**: Dimensiones de input/output no coinciden

### 💡 **Tips**:
- Empieza con curriculum rápido (4 etapas, pocos epochs)
- Ajusta thresholds según task difficulty
- Usa `log_interval=5` para tareas rápidas
- Monitorea gate_diversity para ver si usa todos los bloques

---

## 📝 Notas de Implementación

### Variables Globales Eliminadas ✅
```python
# ❌ Antes (Fase 31-32 early)
from core.reasoning.reasoner_manager import GLOBAL_REASONER

# ✅ Ahora (Fase 33)
manager = CurriculumManager(
    reasoner_manager=reasoner_manager,  # Inyección de dependencia
    graph=graph
)
```

### Integración con CognitiveAppState ✅
```python
# En api/routes/curriculum.py
def get_curriculum_manager(state = Depends(get_app_state)):
    return CurriculumManager(
        reasoner_manager=state.reasoner_manager,
        graph=state.graph
    )
```

---

## 🏆 Conclusión

La Fase 33 implementa un **sistema profesional de Curriculum Learning** que:

- ✅ Entrena el Reasoner progresivamente
- ✅ Sin variables globales (arquitectura limpia)
- ✅ Checkpointing automático
- ✅ Métricas avanzadas (8 diferentes)
- ✅ API REST completa
- ✅ Dashboard interactivo
- ✅ Tests exhaustivos
- ✅ Documentación completa

**El Reasoner ahora puede "aprender a aprender", adaptándose a tareas cada vez más complejas de manera natural y observable.**

---

**Autor**: Neural Core Team  
**Fase**: 33  
**Estado**: ✅ Completo  
**Próximo**: Fase 34 (Benchmark Suite) o Fase 35 (Federated Reasoners)
```
