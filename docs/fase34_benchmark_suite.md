```markdown
# 🧮 Fase 34 - Cognitive Benchmark Suite

## 🎯 Objetivo

Implementar un **sistema profesional de benchmarking científico** que permita comparar configuraciones del Reasoner con:
- ✅ Reproducibilidad total (seeds, provenance, git tracking)
- ✅ Análisis estadístico riguroso (t-tests, CI, effect size)
- ✅ Multi-run aggregation para validez estadística
- ✅ Baselines automáticos para referencia objetiva
- ✅ Reportes multi-formato (MD, HTML, LaTeX, CSV, JSON)
- ✅ Dashboard interactivo
- ✅ API REST completa

---

## 🧠 Concepto Base

El benchmarking científico debe ser:

1. **Reproducible**: Mismo config + mismo seed = mismos resultados
2. **Estadísticamente válido**: Múltiples runs, confidence intervals, p-values
3. **Comparable**: Baselines objetivos, métricas estándar
4. **Auditable**: Provenance completo (git, environment, timestamps)
5. **Publicable**: Reportes listos para papers científicos

```
Workflow típico:
Config → N runs → Aggregated metrics → Statistical comparison → Report
```

---

## 📁 Estructura de Archivos

```
src/core/benchmark/
├── __init__.py                    # Exportaciones del módulo
├── configurations.py              # BenchmarkConfig con reproducibilidad ⭐
├── metrics.py                     # Métricas científicas avanzadas ⭐
├── provenance.py                  # Tracking de reproducibilidad ⭐
├── baseline.py                    # Estrategias baseline ⭐
├── comparator.py                  # Análisis estadístico ⭐
├── benchmark_suite.py             # Runner principal ⭐
└── report_generator.py            # Reportes multi-formato ⭐

src/api/routes/
└── benchmark.py                   # API REST ⭐

dashboard/
└── dashboard_benchmark.py         # Dashboard Streamlit ⭐

examples/
├── benchmark_demo.py              # Demo básico ⭐
└── benchmark_scientific.py        # Demo científico completo ⭐

tests/
└── test_benchmark.py              # Tests exhaustivos ⭐

docs/
└── fase34_benchmark_suite.md      # Esta documentación
```

---

## 🧩 Componentes Principales

### 1️⃣ **BenchmarkConfig (`configurations.py`)**

Configuración completa y reproducible de un experimento.

#### Características:

- ✅ **Hashing único** para identificación
- ✅ **Seed control** para reproducibilidad
- ✅ **Versionado** de configs
- ✅ **Validación** de parámetros
- ✅ **Serialización** JSON/YAML

#### Parámetros Principales:

| Categoría | Parámetros | Descripción |
|-----------|------------|-------------|
| **Identidad** | name, description, version, tags | Metadata |
| **Reproducibilidad** | seed, deterministic | Control de randomness |
| **Reasoner** | reasoner_mode, n_hidden, n_blocks | Arquitectura |
| **Curriculum** | use_curriculum, curriculum_type | Aprendizaje progresivo |
| **Evolution** | evolution_strategy, mutation_scale | Estrategia de optimización |
| **Training** | n_runs, max_epochs_per_stage | Parámetros de entrenamiento |

#### Ejemplo de Uso:

```python
from core.benchmark import BenchmarkConfig

config = BenchmarkConfig(
    name="my_experiment",
    description="Curriculum with top-k gates",
    seed=42,
    use_curriculum=True,
    reasoner_mode="topk",
    topk_value=2,
    n_runs=5,  # Múltiples runs para stats
)

# Hash único
print(config.hash())  # "a3b5c7d9e1f2"

# Guardar
config.to_json("configs/my_experiment.json")
```

#### Configs Pre-Definidas:

```python
from core.benchmark import BENCHMARK_CONFIGS, list_configs

# Listar disponibles
print(list_configs())
# ['baseline_random', 'curriculum_softmax', 'curriculum_topk', ...]

# Obtener config
config = BENCHMARK_CONFIGS["curriculum_softmax"]
```

---

### 2️⃣ **BenchmarkMetrics (`metrics.py`)**

Métricas científicas completas para evaluar performance.

#### Métricas Implementadas (15+):

**Performance**:
- `final_loss`: Loss al final del entrenamiento
- `best_loss`: Mejor loss alcanzado
- `final_accuracy`: Accuracy final
- `best_accuracy`: Mejor accuracy

**Convergencia**:
- `convergence_epoch`: Primera época bajo threshold
- `time_to_threshold`: Tiempo hasta converger
- `converged`: Booleano de convergencia

**Estabilidad**:
- `loss_std`: Desviación estándar del loss
- `loss_variance`: Varianza
- `training_stability`: Score de estabilidad [0, 1]
- `loss_trend_slope`: Pendiente de mejora

**Gates Cognitivos**:
- `gate_diversity`: Uniformidad en uso de bloques
- `gate_entropy`: Entropía de Shannon
- `gate_consistency`: Consistencia temporal
- `gate_utilization`: % de bloques activos
- `dominant_gates`: Bloques más usados

**Eficiencia**:
- `total_epochs`: Total de épocas
- `total_training_time`: Tiempo en segundos
- `epochs_per_second`: Velocidad

**Generalización**:
- `train_loss` / `test_loss`
- `generalization_gap`: test - train
- `overfitting_score`: Grado de overfitting

#### Agregación de Múltiples Runs:

```python
from core.benchmark import BenchmarkMetrics

metrics_list = [run1_metrics, run2_metrics, run3_metrics]

# Agregar con estadísticas
aggregated = BenchmarkMetrics.aggregate(
    metrics_list,
    confidence_level=0.95
)

# Acceder a estadísticas
mean_loss = aggregated.get_mean("final_loss")
std_loss = aggregated.get_std("final_loss")
ci_low, ci_high = aggregated.get_ci("final_loss")

# Formateo automático
formatted = aggregated.format_metric("final_loss", precision=4)
# "0.0234 ± 0.0012 [0.0220, 0.0248]"
```

---

### 3️⃣ **BenchmarkProvenance (`provenance.py`)**

Sistema de rastreo completo para reproducibilidad.

#### Información Capturada:

**Environment**:
- Python version
- NumPy version
- OS, platform, machine
- CPU count

**Git State**:
- Commit hash
- Branch name
- Is dirty (uncommitted changes)
- Remote URL

**Random State**:
- Seed usado
- NumPy random state (serializado)

**Config Completa**:
- Full config JSON

#### Ejemplo de Uso:

```python
from core.benchmark import BenchmarkProvenance, verify_reproducibility

# Capturar provenance
provenance = BenchmarkProvenance.capture(config)

print(provenance.summary())
# Run ID: 20250106_230145_123456
# Config: curriculum_softmax (hash: a3b5c7d9e1f2)
# Timestamp: 2025-01-06T23:01:45
# Python: 3.10.0
# NumPy: 1.24.0
# OS: Linux 5.15.0-91-generic
# Git: main@a3b5c7d9
# Reproducible: ✅

# Verificar si es reproducible en ambiente actual
check = verify_reproducibility(provenance)

if not check["can_reproduce"]:
    for warning in check["warnings"]:
        print(f"⚠️  {warning}")
```

---

### 4️⃣ **Baseline Strategies (`baseline.py`)**

Estrategias de referencia para comparación objetiva.

#### Baselines Disponibles:

| Baseline | Descripción | Uso |
|----------|-------------|-----|
| **random_uniform** | Gates aleatorios uniformes [0, 1] | Baseline más básico |
| **random_softmax** | Gates con softmax aleatorio | Similar a Reasoner sin aprendizaje |
| **equal** | Todos los gates iguales (1/N) | Activación uniforme |
| **binary_random** | Gates binarios (0 o 1) | On/Off aleatorio |
| **topk_random** | Activa K bloques aleatorios | Top-K sin aprendizaje |
| **first_k** | Siempre los primeros K | Estrategia determinística |
| **gaussian** | Distribución gaussiana | Preferencia por bloques centrales |

#### Ejemplo de Uso:

```python
from core.benchmark import BaselineReasoner, evaluate_baseline

# Crear baseline reasoner
baseline = BaselineReasoner(strategy="random_uniform", n_blocks=3)

# Generar gates
state = np.random.rand(10)
gates = baseline.predict(state)

# Evaluar baseline en task
loss, accuracy = evaluate_baseline(
    baseline_strategy="random_uniform",
    graph=graph,
    X=X_train,
    Y=Y_train
)
```

---

### 5️⃣ **BenchmarkComparator (`comparator.py`)**

Análisis estadístico riguroso de resultados.

#### Características:

- ✅ **T-tests** para comparación pareada
- ✅ **Confidence Intervals** (95% default)
- ✅ **Effect Size** (Cohen's d)
- ✅ **Bonferroni correction** para múltiples comparaciones
- ✅ **Friedman test** para múltiples grupos
- ✅ **Ranking** automático

#### Ejemplo de Uso:

```python
from core.benchmark import BenchmarkComparator

comparator = BenchmarkComparator(confidence_level=0.95)

# Comparar dos configs
comparison = comparator.compare_two(
    metrics_a=results_config_a,
    metrics_b=results_config_b,
    metric="final_loss",
    config_name_a="Curriculum",
    config_name_b="Baseline"
)

print(comparison.summary())
# Comparación: Curriculum vs Baseline
# Métrica: final_loss
#
# Config A: 0.0234 ± 0.0012 (median: 0.0230)
# Config B: 0.0456 ± 0.0023 (median: 0.0450)
#
# T-test: t=-12.345, p=0.0001 ✅
# Cohen's d: 0.856 (large)
#
# Winner: A 🏆
# Improvement: 48.7%

# Rankear todas las configs
ranking = comparator.rank_configs(results_dict, metric="final_loss")

for name, mean, std, rank in ranking:
    print(f"{rank}. {name:25s} | {mean:.4f} ± {std:.4f}")
```

---

### 6️⃣ **BenchmarkSuite (`benchmark_suite.py`)**

Runner principal que orquesta todo el proceso.

#### Características:

- ✅ **Reproducibilidad** automática (seeds)
- ✅ **Multi-run** con agregación
- ✅ **Provenance** capture
- ✅ **Auto-save** de resultados
- ✅ **Logging** estructurado

#### Ejemplo de Uso:

**Single Benchmark**:

```python
from core.benchmark import BenchmarkSuite, BENCHMARK_CONFIGS

suite = BenchmarkSuite(
    output_dir="data/benchmarks/results",
    verbose=True
)

result = suite.run_single(
    config=BENCHMARK_CONFIGS["curriculum_softmax"],
    reasoner_manager=reasoner_manager,
    graph=graph,
    save_results=True
)

# Acceder a resultados
print(f"Final loss: {result.metrics.get_mean('final_loss'):.4f}")
print(f"Run ID: {result.provenance.run_id}")
```

**Comparison**:

```python
comparison_report = suite.run_comparison(
    configs=[
        BENCHMARK_CONFIGS["curriculum_softmax"],
        BENCHMARK_CONFIGS["curriculum_topk"],
        BENCHMARK_CONFIGS["baseline_random"],
    ],
    reasoner_manager=reasoner_manager,
    graph=graph,
    metric="final_loss"
)

# Ver ranking
for name, mean, std, rank in comparison_report.ranking:
    print(f"{rank}. {name} | {mean:.4f}")
```

---

### 7️⃣ **ReportGenerator (`report_generator.py`)**

Generador de reportes multi-formato.

#### Formatos Soportados:

1. **Markdown** - Legible, versionable con git
2. **HTML** - Interactivo con tablas
3. **LaTeX** - Para papers científicos
4. **CSV** - Análisis en Excel/Pandas
5. **JSON** - Programático

#### Ejemplo de Uso:

```python
from core.benchmark import ReportGenerator

generator = ReportGenerator()

generator.generate_all(
    report=comparison_report,
    output_dir="data/benchmarks/reports/exp001",
    formats=["markdown", "html", "latex", "csv", "json"]
)

# Genera:
# - report.md
# - report.html
# - report.tex
# - data.csv
# - data.json
```

---

## 🌐 API REST

### Endpoints Disponibles:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/benchmark/configs` | Lista configs disponibles |
| GET | `/benchmark/config/{name}` | Detalles de una config |
| POST | `/benchmark/run` | Ejecuta un benchmark |
| POST | `/benchmark/compare` | Ejecuta comparación |
| GET | `/benchmark/status` | Estado actual |
| GET | `/benchmark/results` | Lista todos los resultados |
| GET | `/benchmark/result/{run_id}` | Detalle de un resultado |
| GET | `/benchmark/reports` | Lista reportes generados |
| DELETE | `/benchmark/results` | Limpia resultados |

### Ejemplos de Uso:

#### Listar Configs:

```bash
curl http://localhost:8000/benchmark/configs
```

#### Ejecutar Benchmark:

```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"config_name": "curriculum_softmax", "save_results": true}'
```

#### Ejecutar Comparación:

```bash
curl -X POST http://localhost:8000/benchmark/compare \
  -H "Content-Type: application/json" \
  -d '{
    "config_names": ["curriculum_softmax", "curriculum_topk", "baseline_random"],
    "metric": "final_loss"
  }'
```

#### Ver Resultados:

```bash
curl http://localhost:8000/benchmark/results | jq
```

---

## 🎨 Dashboard Streamlit

### Lanzar Dashboard:

```bash
PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py
```

**Accede**: http://localhost:8504

### Características:

**3 Modos de Operación**:

1. **📊 Ver Resultados**
   - Tabla de todos los resultados
   - Gráfico comparativo
   - Ver detalles completos de runs

2. **🚀 Ejecutar Benchmark**
   - Selector de configs
   - Preview de parámetros
   - Ejecución con un click

3. **⚖️ Comparar Configs**
   - Selección múltiple de configs
   - Elección de métrica
   - Reportes automáticos

**Features**:
- Auto-refresh durante ejecución
- Métricas en tiempo real
- Tabla interactiva
- Gráficos Plotly
- Export de datos

---

## 🚀 Guía de Uso

### Opción 1: Demo Básico

```bash
PYTHONPATH=src python examples/benchmark_demo.py
```

### Opción 2: Demo Científico

```bash
PYTHONPATH=src python examples/benchmark_scientific.py
```

### Opción 3: Con Servidor + Dashboard

```bash
# Terminal 1: Servidor
PYTHONPATH=src uv run uvicorn api.server:app --reload

# Terminal 2: Dashboard
PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py

# Terminal 3: Ejecutar benchmark
curl -X POST http://localhost:8000/benchmark/run \
  -d '{"config_name": "curriculum_fast"}'
```

### Opción 4: Programático

```python
from core.benchmark import BenchmarkSuite, BENCHMARK_CONFIGS

suite = BenchmarkSuite()

result = suite.run_single(
    BENCHMARK_CONFIGS["curriculum_softmax"],
    reasoner_manager,
    graph
)

print(result.metrics.format_metric("final_loss"))
```

---

## 📊 Casos de Uso

### Caso 1: Comparar Curriculum vs Baseline

```python
from core.benchmark import BenchmarkSuite, BENCHMARK_CONFIGS

suite = BenchmarkSuite()

report = suite.run_comparison(
    configs=[
        BENCHMARK_CONFIGS["curriculum_softmax"],
        BENCHMARK_CONFIGS["baseline_random"],
    ],
    reasoner_manager=reasoner_manager,
    graph=graph,
    metric="final_loss"
)

# ¿Curriculum aprende mejor que random?
winner = report.ranking[0][0]
print(f"Winner: {winner}")
```

### Caso 2: Optimizar Hyperparámetros

```python
from core.benchmark import create_custom_config

# Probar diferentes mutation scales
configs = [
    create_custom_config(name=f"mutation_{scale}", mutation_scale=scale)
    for scale in [0.01, 0.03, 0.05, 0.1]
]

report = suite.run_comparison(configs, reasoner_manager, graph)

# Ver cuál funciona mejor
best_config = report.ranking[0][0]
```

### Caso 3: Validar Reproducibilidad

```python
# Run 1
result1 = suite.run_single(config, reasoner_manager, graph)

# Run 2 con mismo seed
result2 = suite.run_single(config, reasoner_manager, graph)

# ¿Resultados idénticos?
loss1 = result1.metrics.get_mean("final_loss")
loss2 = result2.metrics.get_mean("final_loss")

assert abs(loss1 - loss2) < 1e-6, "Not reproducible!"
```

---

## 🎯 Mejores Prácticas

### 1. Siempre usar N runs >= 3

```python
config = BenchmarkConfig(
    name="my_exp",
    n_runs=5,  # ✅ Mínimo 3, ideal 5-10
)
```

### 2. Controlar seeds para reproducibilidad

```python
config = BenchmarkConfig(
    name="my_exp",
    seed=42,  # ✅ Seed fijo
    deterministic=True,
)
```

### 3. Incluir baselines

```python
configs = [
    my_custom_config,
    BENCHMARK_CONFIGS["baseline_random"],  # ✅ Baseline
]
```

### 4. Usar configs rápidas para testing

```python
# Testing
config_test = BENCHMARK_CONFIGS["curriculum_fast"]

# Production
config_prod = BENCHMARK_CONFIGS["curriculum_softmax"]
```

### 5. Guardar provenance

```python
result = suite.run_single(config, reasoner_manager, graph, save_results=True)

# ✅ Provenance guardado automáticamente
print(result.provenance.is_reproducible())
```

### 6. Generar reportes completos

```python
generator = ReportGenerator()
generator.generate_all(
    report,
    output_dir="reports/exp001",
    formats=["markdown", "html", "latex", "csv", "json"]
)
```

---

## 🧪 Testing

### Ejecutar Tests:

```bash
# Todos los tests
pytest tests/test_benchmark.py -v

# Test específico
pytest tests/test_benchmark.py::test_config_hash -v

# Con coverage
pytest tests/test_benchmark.py --cov=src/core/benchmark --cov-report=html
```

### Tests Implementados (30+):

- ✅ BenchmarkConfig: creación, hash, serialización, validación
- ✅ BenchmarkMetrics: agregación, CI, formateo
- ✅ Provenance: captura, reproducibilidad
- ✅ Baselines: generación de gates, BaselineReasoner
- ✅ Comparator: t-tests, ranking, effect size
- ✅ Helper functions: stability, trend, consistency
- ✅ Integración: workflow completo

---

## 🏆 Beneficios del Sistema

### 1️⃣ **Validez Científica**
- Multiple runs con agregación estadística
- Confidence intervals al 95%
- P-values para significancia
- Effect sizes (Cohen's d)

### 2️⃣ **Reproducibilidad Total**
- Seeds controlados
- Provenance completo (git, env, timestamps)
- Verificación automática
- State serialization

### 3️⃣ **Comparabilidad**
- Baselines objetivos
- Métricas estándar
- Statistical tests
- Ranking automático

### 4️⃣ **Publicabilidad**
- Reportes LaTeX para papers
- HTML interactivo
- CSV para análisis
- JSON programático

### 5️⃣ **Automatización**
- API REST completa
- Dashboard interactivo
- CLI friendly
- CI/CD ready

---

## 📚 Próximos Pasos Sugeridos

Después de dominar la Fase 34:

### Fase 35: **Federated Reasoners**
- Conectar Orange Pi + Cloud
- Sincronización de experimentos
- Benchmark distribuido
- Agregación de resultados federados

### Fase 36: **AutoML for Reasoners**
- Optimización automática de hyperparámetros
- Neural Architecture Search para Reasoner
- Meta-learning
- Transfer learning entre tasks

### Fase 37: **Benchmark Suite Extensions**
- More baselines (random forest, SVM, etc.)
- Cross-validation
- Bayesian optimization
- Wandb/MLflow integration

---

## 🎓 Comparación con Fase 33

| Aspecto | Fase 33 (Curriculum) | Fase 34 (Benchmark Suite) |
|---------|---------------------|---------------------------|
| **Objetivo** | Entrenar progresivamente | Comparar científicamente |
| **Output** | Reasoner entrenado | Análisis estadístico |
| **Reproducibilidad** | Checkpoints | Provenance completo + seeds |
| **Comparación** | Historial de etapas | T-tests, CI, effect size |
| **Baselines** | No | 8 baselines automáticos |
| **Reportes** | Dashboard | 5 formatos (MD/HTML/LaTeX/CSV/JSON) |
| **Multi-run** | Single run | N runs con agregación |
| **Validez científica** | Observacional | Statistical rigor |

**Relación**: Fase 33 entrena, Fase 34 valida y compara.

---

## 🔥 Conclusión

La Fase 34 implementa un **sistema de benchmarking de nivel científico** que:

- ✅ **Valida científicamente** el aprendizaje curriculum (Fase 33)
- ✅ **Compara** diferentes estrategias con rigor estadístico
- ✅ **Reproduce** experimentos exactamente
- ✅ **Publica** resultados en formatos académicos
- ✅ **Automatiza** todo el pipeline de benchmarking

**Ahora puedes responder preguntas como**:
- ¿Curriculum learning es significativamente mejor que random?
- ¿Qué configuración de gates (softmax vs top-k) funciona mejor?
- ¿Los resultados son reproducibles en otro ambiente?
- ¿Qué tan grande es el effect size?

**El sistema está listo para**:
- Papers científicos (reportes LaTeX)
- Análisis de datos (CSV export)
- Automatización (API + CLI)
- Integración continua (reproducibility checks)

---

**Autor**: Neural Core Team  
**Fase**: 34  
**Estado**: ✅ Completo  
**Próximo**: Fase 35 (Federated Reasoners) o Fase 36 (AutoML)
```
