# 🤖 Fase 35 - Agentic Reasoners (MVP Día 1)

## 🎯 Objetivo

Implementar un **sistema agentivo completo** inspirado en Claude Agent SDK y ReAct que permite al Reasoner razonar, planificar y actuar de forma autónoma con el loop **Plan-Act-Reflect**.

## 🧠 Concepto: Loop Agentivo

```
CONTEXT  → Recopila información del sistema
   ↓
PLAN     → Genera plan de acciones
   ↓
ACT      → Ejecuta acciones con tools
   ↓
VERIFY   → Verifica calidad de resultados
   ↓
REFLECT  → Aprende de la experiencia
   ↓
  Loop   ← Repite hasta lograr objetivo
```

---

## 📁 Estructura

```
src/core/agents/
├── base.py              # BaseAgent + telemetría
├── context_agent.py     # Recopilación contexto
├── planner_agent.py     # Planificación
├── action_agent.py      # Ejecución
├── verifier_agent.py    # Verificación
├── reflector_agent.py   # Reflexión
├── orchestrator.py      # Coordinador
├── memory.py            # Sistema memoria
└── __init__.py          # Factory

src/core/tools/
├── base.py              # BaseTool
├── cognitive_tools.py   # 5 tools
├── registry.py          # ToolRegistry
└── __init__.py          # Factory

examples/agentic_demo.py # Demo funcional
```

---

## 🧩 Componentes

### 1. BaseAgent
**Ubicación**: `src/core/agents/base.py`

Abstracción base con telemetría automática.

**Clases**:
- `AgentAction`: Acción a ejecutar (tool, params, reasoning, priority)
- `AgentObservation`: Resultado de acción (result, success, error)
- `AgentThought`: Pensamiento del agente
- `BaseAgent`: Clase base abstracta
- `AgentRegistry`: Registro global de agentes

**Stats automáticas**: call_count, success_rate, avg_time

### 2. ContextAgent
**Ubicación**: `src/core/agents/context_agent.py`

Recopila estado del grafo, reasoner y detecta issues.

**Recopila**:
- Graph: bloques, conexiones, gates
- Reasoner: mode, estado, historial
- Síntesis: complejidad, análisis gates
- Issues: low_activation, high_variance, reasoner_unavailable

### 3. PlannerAgent
**Ubicación**: `src/core/agents/planner_agent.py`

Genera planes con 4 estrategias.

**Estrategias**:
- `optimize_performance`: Maximizar performance
- `explore`: Explorar configuraciones
- `learn`: Curriculum learning
- `diagnose`: Diagnosticar problemas

**Output**: plan (lista AgentAction), reasoning, confidence

### 4. ActionAgent
**Ubicación**: `src/core/agents/action_agent.py`

Ejecuta acciones con ToolRegistry.

**Features**:
- Integración con ToolRegistry
- Reintentos automáticos (max 2)
- Backoff exponencial
- Observaciones estructuradas

### 5. VerifierAgent
**Ubicación**: `src/core/agents/verifier_agent.py`

Verifica calidad con scoring fuzzy.

**Criterios** (weights):
- Performance: 40% (loss, accuracy)
- Stability: 30% (success rate)
- Efficiency: 20% (sin errores)
- Novelty: 10% (exploración)

**Decisiones**:
- score ≥ 0.75 → `accept`
- score ≥ 0.50 → `retry`
- score < 0.50 → `abort`

**LLM-ready**: Preparado para LLM-as-Judge (Día 2)

### 6. ReflectorAgent
**Ubicación**: `src/core/agents/reflector_agent.py`

Reflexiona y aprende de experiencias.

**Genera**:
- Insights de alto nivel
- Patrones success/failure
- Aprendizajes clave
- Recomendaciones futuras
- Decisión de actualizar reasoner

### 7. CognitiveOrchestrator
**Ubicación**: `src/core/agents/orchestrator.py`

Coordina el loop completo.

**Ciclo**:
1. Context → Plan → Act → Verify → Reflect
2. Early stopping si `decision == "accept"`
3. Historial completo de ciclos
4. Stats agregadas de todos los agentes

### 8. AgentMemory
**Ubicación**: `src/core/agents/memory.py`

Sistema de memoria episódica y semántica.

**Tipos**:
- Episódica: Historial de ciclos (Episode)
- Semántica: Conocimiento acumulado (KnowledgeEntry)
- Persistencia: save/load JSON

---

## 🔧 Tool System

### BaseTool + ToolRegistry
**Ubicación**: `src/core/tools/`

Registry centralizado con 5 tools implementados:

| Tool | Descripción |
|------|-------------|
| `reasoner_evolve` | Evoluciona el reasoner |
| `graph_analyze` | Analiza estructura del grafo |
| `curriculum_start` | Inicia curriculum learning |
| `benchmark_quick` | Ejecuta benchmark rápido |
| `system_health_check` | Verifica salud del sistema |

---

## 🚀 Uso Rápido

### Demo Funcional

```bash
PYTHONPATH=src python examples/agentic_demo.py
```

### Código Mínimo

```python
from core.agents import create_default_orchestrator
from core.tools import create_default_registry

# Setup
orchestrator = create_default_orchestrator(
    graph=graph,
    reasoner_manager=reasoner_manager,
    goal="optimize_performance",
    verbose=True,
)

tool_registry = create_default_registry(graph, reasoner_manager)
orchestrator.action_agent.tool_registry = tool_registry

# Ejecutar loop
result = await orchestrator.loop(
    max_iterations=3,
    goal="optimize_performance",
    early_stop=True,
)

# Resultado
print(f"Completado: {result['success']}")
print(f"Ciclos: {result['iterations_run']}")
print(f"Decisión: {result['final_decision']}")
```

---

## 📊 Casos de Uso

### 1. Optimización Automática
```python
goal = "optimize_performance"
# → Evoluciona reasoner, ejecuta benchmarks, analiza mejoras
```

### 2. Exploración
```python
goal = "explore"
# → Prueba diferentes configs, modos, estrategias
```

### 3. Aprendizaje Progresivo
```python
goal = "learn"
# → Inicia curriculum, monitorea progreso, ajusta
```

### 4. Diagnóstico
```python
goal = "diagnose"
# → Health checks, validaciones, identifica issues
```

---

## 🔬 Integración con Fases Anteriores

| Fase | Integración |
|------|-------------|
| **31-32** (Reasoner) | ContextAgent lee estado, ActionAgent evoluciona |
| **33** (Curriculum) | CurriculumStartTool lo inicia automáticamente |
| **34** (Benchmark) | BenchmarkQuickTool ejecuta evaluaciones |

**Sin cambios breaking** - Todo backward compatible.

---

## 📈 Telemetría

### Nivel Agente
```python
stats = agent.get_stats()
# {call_count, success_rate, avg_time, thoughts_count}
```

### Nivel Tool
```python
stats = tool.get_stats()
# {call_count, success_rate, avg_time}
```

### Nivel Orchestrator
```python
stats = orchestrator.get_stats()
# {cycles, accepts, retries, aborts, avg_score, agents: {...}}
```

---

## 🎯 Mejores Prácticas

1. **Seleccionar goal apropiado** según objetivo
2. **Configurar early_stop=True** para eficiencia
3. **Usar memory system** para acumular experiencia
4. **Monitorear telemetría** para identificar cuellos de botella
5. **Extender con nuevos tools** según necesidad

---

## 🎨 Dashboard Hub (Integración Completa)

Para ver **todos los dashboards del proyecto en una sola aplicación**:

```bash
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
```

**Features**:
- 🤖 **Agentic Loop**: Sistema agentivo (Fase 35)
- 📊 **Benchmark Suite**: Evaluaciones científicas (Fase 34)
- 📚 **Curriculum Learning**: Entrenamiento progresivo (Fase 33)
- 🧠 **Reasoner Control**: Control del razonador (Fase 32)
- 🔴 **Live Stream**: Visualización en tiempo real
- 📈 **PyG Visualization**: Grafos interactivos

**Ventajas del Hub**:
- Vista unificada de todo el sistema
- Navegación con pestañas
- Estado del API server en tiempo real
- Enlaces rápidos a dashboards individuales

---

## 🔜 Roadmap Día 2

### Pendiente:

1. **LLM Integration**
   - LLM client abstraction (`src/core/llm/base.py`)
   - Gemini client (`gemini_client.py`)
   - DeepSeek client (`deepseek_client.py`)
   - LLM-as-Judge en VerifierAgent
   - Prompt templates

2. **API REST**
   - `/agents/run-loop`
   - `/agents/status`
   - `/agents/history`
   - `/agents/stats`

3. **Dashboard Agentic Completo**
   - Visualización del loop en tiempo real
   - Control interactivo de agentes
   - Métricas de performance
   - Historial de ciclos

4. **Tests**
   - Tests unitarios de agentes
   - Tests de tools
   - Tests de integración

---

## 🏆 Beneficios

✅ **Autonomía**: El Reasoner actúa sin intervención  
✅ **Aprendizaje**: Acumula experiencia y mejora  
✅ **Integración**: Usa todo lo existente (Fases 31-34)  
✅ **Extensibilidad**: Fácil añadir agentes/tools  
✅ **Observabilidad**: Telemetría completa  
✅ **LLM-ready**: Preparado para Gemini/DeepSeek  

---

## 📚 Referencias

- **Claude Agent SDK**: Loop agentivo con tools
- **ReAct Paper** (Yao et al. 2022): Reasoning + Acting
- **Neuraxon** (HuggingFace): LLM-as-Judge inspiration

---

**Autor**: Neural Core Team  
**Fase**: 35 (MVP Día 1 ✅)  
**Estado**: Funcional - Listo para Día 2  
**Próximo**: LLM Integration + API + Dashboard
