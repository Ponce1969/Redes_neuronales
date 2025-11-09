# 📊 Dashboard System

Sistema completo de dashboards Streamlit para Neural Core.

---

## 🎯 Dashboard Hub (Recomendado)

**Centro de control unificado** con todos los dashboards integrados:

```bash
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
```

**Features**:
- 6 pestañas integradas
- Vista unificada del sistema
- Estado del API server
- Enlaces a dashboards individuales

---

## 📁 Dashboards Disponibles

### **Fase 35 - Agentic Reasoners** 🤖
- **dashboard_hub.py** ⭐ - Hub central (todos los dashboards)
- *(dashboard_agentic.py pendiente para Día 2)*

### **Fase 34 - Benchmark Suite** 📊
```bash
PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py
```
- Ver resultados de benchmarks
- Ejecutar nuevos benchmarks
- Comparar configuraciones
- Gráficos científicos

### **Fase 33 - Curriculum Learning** 📚
```bash
PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py
```
- Monitor de curriculum learning
- Progreso por etapa
- Métricas en tiempo real
- Control de entrenamiento

### **Fase 32 - Reasoner Control** 🧠
```bash
# Panel de control
PYTHONPATH=src streamlit run dashboard/dashboard_reasoner_panel.py

# Con visualización PyG
PYTHONPATH=src streamlit run dashboard/dashboard_pyg_with_reasoner.py
```
- Control del reasoner
- Predicción de gates
- Evolución
- Visualización interactiva

### **Visualización Avanzada** 📈

**Live Stream**:
```bash
PYTHONPATH=src streamlit run dashboard/dashboard_live_stream.py
```
- Actualización automática
- Métricas en tiempo real

**PyG Visualization**:
```bash
# Interactivo
PYTHONPATH=src streamlit run dashboard/dashboard_pyg_interactive.py

# Básico
PYTHONPATH=src streamlit run dashboard/dashboard_pyg_viz.py
```
- Grafos 3D interactivos
- Análisis de conectividad

---

## 🚀 Inicio Rápido

### 1. Dashboard Hub (Opción Más Fácil)

```bash
# Solo el hub
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
```

### 2. Setup Completo (API + Dashboard)

**Terminal 1** - Servidor API:
```bash
PYTHONPATH=src uv run uvicorn api.server:app --reload
```

**Terminal 2** - Dashboard Hub:
```bash
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
```

### 3. Dashboards Individuales

Ejecuta cualquier dashboard individual según necesidad (ver secciones arriba).

---

## 📖 Organización

```
dashboard/
├── README.md                           # Esta guía
├── dashboard_hub.py                    # 🎯 Hub central ⭐
├── dashboard_agentic.py                # 🤖 (Día 2)
├── dashboard_benchmark.py              # 📊 Fase 34
├── dashboard_curriculum.py             # 📚 Fase 33
├── dashboard_reasoner_panel.py         # 🧠 Fase 32
├── dashboard_pyg_with_reasoner.py      # 🧠 Fase 32 (con PyG)
├── dashboard_live_stream.py            # 🔴 Live
├── dashboard_pyg_interactive.py        # 📈 PyG avanzado
└── dashboard_pyg_viz.py                # 📈 PyG básico
```

---

## 💡 Tips

1. **Usa el Hub** para tener todo en un solo lugar
2. **Inicia el API server** para funcionalidad completa
3. **Dashboards individuales** para sesiones largas enfocadas
4. **Live Stream** para monitoreo continuo
5. **PyG dashboards** para análisis profundo de grafos

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'core'"
```bash
# Asegúrate de usar PYTHONPATH=src
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
```

### "Connection error to API server"
```bash
# Inicia el servidor API primero
PYTHONPATH=src uv run uvicorn api.server:app --reload
```

### "Port already in use"
```bash
# Streamlit usa puerto 8501 por defecto
# Para cambiar puerto:
PYTHONPATH=src streamlit run dashboard/dashboard_hub.py --server.port 8502
```

---

## 📚 Documentación

- **Fase 35**: `docs/fase35_agentic_reasoners.md`
- **Fase 34**: `docs/fase34_benchmark_suite.md`
- **Fase 33**: `docs/fase33_curriculum_learning.md`
- **Fase 32**: `docs/fase32_reasoner_integration.md`

---

**Neural Core Dashboard System v1.0**
