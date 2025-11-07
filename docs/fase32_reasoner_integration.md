# 🧠 Fase 32: Reasoner Integration & Dashboard

## 📋 Descripción

La **Fase 32** integra completamente el Reasoner con la API REST, dashboards y sistema de persistencia, haciendo que las decisiones del Reasoner sean observables y controlables en tiempo real.

## 🎯 Componentes Implementados

### 1. ReasonerManager (`src/core/reasoning/reasoner_manager.py`)

Controlador centralizado thread-safe del Reasoner:

- **Gestión de decisiones**: Calcula gates en tiempo real con locks para thread-safety
- **Evolución asíncrona**: Ejecuta evolución en background sin bloquear el servidor
- **Persistencia eficiente**: Guarda/carga estado en formato `.npz` comprimido
- **Historial de gates**: Mantiene últimos 100 gates para análisis

```python
from core.reasoning import ReasonerManager

# Inicialización
manager = ReasonerManager(n_inputs=32, n_hidden=64, n_blocks=4, seed=42)

# Decisión de gates
gates = manager.decide(z_per_block, mode="softmax")

# Evolución en background
manager.evolve_async(eval_fn, generations=50, pop_size=8)

# Persistencia
manager.save("data/reasoner_state")
manager.load("data/reasoner_state")
```

### 2. API REST Completa (`src/api/routes/reasoner.py`)

Endpoints para control total del Reasoner:

#### `GET /reasoner/status`
Obtiene estado actual: running, generation, best_loss, etc.

```bash
curl http://localhost:8000/reasoner/status
```

#### `GET /reasoner/gates?n=10`
Obtiene últimos N gates calculados

```bash
curl http://localhost:8000/reasoner/gates?n=5
```

#### `POST /reasoner/predict`
Calcula gates para vectores latentes específicos

```bash
curl -X POST http://localhost:8000/reasoner/predict \
  -H "Content-Type: application/json" \
  -d '{
    "z_vectors": [[0.1, 0.2], [0.3, 0.4]],
    "mode": "softmax",
    "temp": 1.0
  }'
```

#### `POST /reasoner/evolve`
Inicia evolución en background con evaluación real en el grafo

```bash
curl -X POST http://localhost:8000/reasoner/evolve \
  -H "Content-Type: application/json" \
  -d '{
    "generations": 50,
    "pop_size": 10,
    "mutation_scale": 0.03
  }'
```

#### `POST /reasoner/evolve/stop`
Detiene la evolución en curso

```bash
curl -X POST http://localhost:8000/reasoner/evolve/stop
```

#### `POST /reasoner/save`
Guarda estado actual del Reasoner

```bash
curl -X POST http://localhost:8000/reasoner/save
```

#### `POST /reasoner/load`
Carga estado desde disco

```bash
curl -X POST http://localhost:8000/reasoner/load
```

### 3. Integración con CognitiveAppState

El ReasonerManager se inicializa automáticamente al arrancar el servidor:

```python
# En src/api/dependencies.py
app_state = CognitiveAppState(society)

# Inicialización automática con dimensiones del grafo
sample_graph = society.agents[0].graph
n_blocks = len(sample_graph.blocks)
n_inputs = 8 * n_blocks  # Estimación conservadora

app_state.reasoner_manager = ReasonerManager(
    n_inputs=n_inputs,
    n_hidden=n_inputs * 2,
    n_blocks=n_blocks,
    seed=42
)
```

### 4. PersistenceManager Extendido

Ahora guarda/carga el Reasoner automáticamente:

```python
# En src/core/persistence/persistence_manager.py
class PersistenceManager:
    def __init__(self, society, reasoner_manager=None):
        self.society = society
        self.reasoner_manager = reasoner_manager
    
    def save_all(self):
        # Guarda agentes
        for agent in self.society.agents:
            save_weights(agent)
            save_memory(agent)
        
        # Guarda Reasoner
        if self.reasoner_manager:
            self.reasoner_manager.save("data/persistence/reasoner_state")
```

### 5. Dashboard de Control (`dashboard/dashboard_reasoner_panel.py`)

Panel Streamlit interactivo con:

- **Visualización en tiempo real**: Gráfico de barras con gates por bloque
- **Control de evolución**: Botones para start/stop con configuración de parámetros
- **Métricas**: Best loss, generación actual, progreso, estadísticas
- **Persistencia**: Guardar/cargar estado con un click
- **Auto-refresh configurable**: Actualización cada 1-10 segundos

```bash
# Lanzar dashboard
PYTHONPATH=src uv run streamlit run dashboard/dashboard_reasoner_panel.py
```

### 6. Dashboard PyG + Reasoner (`dashboard/dashboard_pyg_with_reasoner.py`)

Visualización integrada que combina:

- **Grafo PyG interactivo**: Nodos movibles, zoom, hover con detalles
- **Coloreo por gates**: Opción para colorear por plan latente o gate del Reasoner
- **Métricas por bloque**: Activación, plan latente, GAT output, gate
- **Tabla detallada**: Todas las features en formato tabular
- **Actualización en tiempo real**: Consume gates de la API automáticamente

```bash
# Lanzar dashboard integrado
PYTHONPATH=src uv run streamlit run dashboard/dashboard_pyg_with_reasoner.py
```

## 🚀 Flujo de Uso Completo

### Setup Inicial

```bash
# Terminal 1: Arrancar servidor con Reasoner integrado
cd /home/gonzapython/Documentos/Redes_Neuronales/neural_core
PYTHONPATH=src uv run uvicorn api.server:app --reload
```

El servidor automáticamente:
1. Inicializa ReasonerManager con dimensiones del grafo
2. Intenta cargar estado previo desde `data/persistence/reasoner_state.npz`
3. Expone todos los endpoints REST en `/reasoner/*`

### Uso desde Dashboard

```bash
# Terminal 2: Lanzar panel de control
PYTHONPATH=src uv run streamlit run dashboard/dashboard_reasoner_panel.py
```

En el dashboard puedes:
- Ver estado actual del Reasoner
- Iniciar evolución con parámetros personalizados
- Monitorear progreso en tiempo real
- Ver gates aplicados a cada bloque
- Guardar mejor configuración encontrada

### Uso desde API (scripts/notebooks)

```python
import requests

API = "http://localhost:8000/reasoner"

# Iniciar evolución
response = requests.post(f"{API}/evolve", json={
    "generations": 100,
    "pop_size": 12,
    "mutation_scale": 0.025
})

# Monitorear progreso
import time
while True:
    status = requests.get(f"{API}/status").json()
    print(f"Gen {status['generation']}/{status['total_generations']}: Loss={status['best_loss']:.4f}")
    
    if not status['running']:
        break
    
    time.sleep(2)

# Guardar mejor Reasoner
requests.post(f"{API}/save")
```

### Visualización Integrada

```bash
# Terminal 3: Dashboard con grafo + gates
PYTHONPATH=src uv run streamlit run dashboard/dashboard_pyg_with_reasoner.py
```

Muestra en tiempo real:
- Nodos del grafo coloreados por gate del Reasoner
- Intensidad del plan latente de cada bloque
- Salida del razonador GAT
- Actualización sincronizada con evolución

## 📊 Ejemplo de Sesión

```bash
# 1. Arrancar servidor
PYTHONPATH=src uv run uvicorn api.server:app --reload
# Output: [ReasonerManager] Inicializado: 3 bloques, 24 inputs, 48 hidden

# 2. En otro terminal, usar curl para evolucionar
curl -X POST http://localhost:8000/reasoner/evolve \
  -H "Content-Type: application/json" \
  -d '{"generations": 50, "pop_size": 10, "mutation_scale": 0.03}'
# Output: {"started":true,"generations":50,"pop_size":10,"mutation_scale":0.03}

# 3. Monitorear estado
watch -n 2 'curl -s http://localhost:8000/reasoner/status | jq'

# 4. Cuando termine, guardar
curl -X POST http://localhost:8000/reasoner/save
# Output: {"saved":true,"path":"data/persistence/reasoner_state.npz"}
```

## 🔧 Configuración Avanzada

### Ajustar Dimensiones del Reasoner

Si tus bloques tienen planes latentes muy grandes:

```python
# En src/api/dependencies.py, modificar:
max_plan_dim = 16  # En vez de 8
n_inputs = max_plan_dim * n_blocks
n_hidden = n_inputs * 3  # Más capacidad
```

### Cambiar Dataset de Evaluación

Por defecto usa XOR. Para cambiar:

```python
# En src/api/routes/reasoner.py, función evaluate_on_graph:
X = np.array([[...]], dtype=np.float32)  # Tu dataset
Y = np.array([...], dtype=np.float32)     # Tus targets
```

### Persistencia Manual

```python
from core.reasoning import ReasonerManager

manager = ReasonerManager(32, 64, 4)

# Entrenar/evolucionar...

# Guardar en ubicación custom
manager.save("/path/to/my_reasoner")

# Cargar después
manager.load("/path/to/my_reasoner")
```

## 🎯 Beneficios de la Integración

### Antes (Fase 31-B)
- Reasoner funcional pero aislado
- Evolución solo en scripts standalone
- Sin visibilidad en tiempo real
- Persistencia manual

### Ahora (Fase 32)
- ✅ **Control REST**: API completa para integración
- ✅ **Visibilidad total**: Dashboards en tiempo real
- ✅ **Auto-persistencia**: Se guarda/carga automáticamente
- ✅ **Evolución integrada**: Usa el grafo activo real
- ✅ **Thread-safe**: Múltiples clientes concurrentes
- ✅ **Background tasks**: No bloquea servidor

## 🔮 Próximos Pasos Sugeridos

### A Corto Plazo

1. **Integrar con Scheduler**: Evolución periódica automática
2. **WebSocket para gates**: Stream continuo de gates
3. **Métricas históricas**: Guardar evolución de loss en DB
4. **Multi-Reasoner**: Ensemble de reasoners que votan

### A Medio Plazo

5. **Reasoner PyTorch**: Versión diferenciable end-to-end
6. **Meta-Learning**: Reasoner que aprende de múltiples tareas
7. **Transfer Learning**: Compartir reasoners entre grafos
8. **Cuantización**: Optimizar para deployment en Orange Pi

## 📁 Archivos Nuevos/Modificados

### Nuevos
```
src/core/reasoning/reasoner_manager.py          # Manager centralizado
src/api/routes/reasoner.py                      # API endpoints
dashboard/dashboard_reasoner_panel.py            # Panel de control
dashboard/dashboard_pyg_with_reasoner.py         # Visualización integrada
docs/fase32_reasoner_integration.md              # Esta documentación
```

### Modificados
```
src/core/reasoning/__init__.py                   # Exporta ReasonerManager
src/api/dependencies.py                          # Inicializa ReasonerManager
src/api/server.py                                # Incluye router del Reasoner
src/core/persistence/persistence_manager.py      # Guarda/carga Reasoner
```

## ✅ Verificación de Instalación

```bash
# 1. Servidor arranca sin errores
PYTHONPATH=src uv run uvicorn api.server:app --reload
# Debe mostrar: [ReasonerManager] Inicializado: X bloques...

# 2. Endpoint responde
curl http://localhost:8000/reasoner/status
# Debe devolver JSON con "running", "best_loss", etc.

# 3. Dashboard funciona
PYTHONPATH=src uv run streamlit run dashboard/dashboard_reasoner_panel.py
# Debe abrir en http://localhost:8501
```

---

**Implementación completada**: Fase 32 ✅  
**Integración**: 100% con sistema existente  
**Estado**: Producción ready  
**Fecha**: Noviembre 2024
