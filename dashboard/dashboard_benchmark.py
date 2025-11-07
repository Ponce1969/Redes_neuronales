"""
Dashboard Streamlit para Benchmarking.

Interfaz interactiva para ejecutar y analizar benchmarks científicos.

Uso:
    PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List

# Configuración de la página
st.set_page_config(
    page_title="🧮 Benchmark Suite",
    page_icon="📊",
    layout="wide",
)

# URL de la API
API_URL = "http://localhost:8000"

# ============================================================================
# Funciones Helper
# ============================================================================

def get_available_configs() -> List[str]:
    """Obtiene lista de configuraciones disponibles."""
    try:
        response = requests.get(f"{API_URL}/benchmark/configs", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return []


def get_config_details(config_name: str) -> Dict[str, Any]:
    """Obtiene detalles de una configuración."""
    try:
        response = requests.get(f"{API_URL}/benchmark/config/{config_name}", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return {}


def run_benchmark(config_name: str) -> bool:
    """Inicia un benchmark."""
    try:
        response = requests.post(
            f"{API_URL}/benchmark/run",
            json={"config_name": config_name, "save_results": True},
            timeout=5,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False


def run_comparison(config_names: List[str], metric: str = "final_loss") -> bool:
    """Ejecuta comparación."""
    try:
        response = requests.post(
            f"{API_URL}/benchmark/compare",
            json={"config_names": config_names, "metric": metric},
            timeout=5,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False


def get_benchmark_status() -> Dict[str, Any]:
    """Obtiene estado del benchmark."""
    try:
        response = requests.get(f"{API_URL}/benchmark/status", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return {}


def get_results() -> List[Dict[str, Any]]:
    """Obtiene lista de resultados."""
    try:
        response = requests.get(f"{API_URL}/benchmark/results", timeout=2)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return []


def get_result_detail(run_id: str) -> Dict[str, Any]:
    """Obtiene detalles de un resultado."""
    try:
        response = requests.get(f"{API_URL}/benchmark/result/{run_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return {}


# ============================================================================
# Header
# ============================================================================

st.title("🧮 Cognitive Benchmark Suite")
st.markdown(
    "Benchmarking científico con reproducibilidad total y análisis estadístico riguroso"
)

st.divider()

# ============================================================================
# Sidebar - Configuración
# ============================================================================

st.sidebar.header("⚙️ Configuración")

# Modo
mode = st.sidebar.radio(
    "Modo de operación",
    ["📊 Ver Resultados", "🚀 Ejecutar Benchmark", "⚖️ Comparar Configs"],
    index=0,
)

st.sidebar.divider()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Intervalo (s)", 1, 10, 3)

# ============================================================================
# Estado del Sistema
# ============================================================================

status = get_benchmark_status()

if not status:
    st.warning("⚠️  No se puede conectar a la API. ¿Está el servidor corriendo?")
    st.code("PYTHONPATH=src uv run uvicorn api.server:app --reload")
    st.stop()

# Métricas del sistema
col1, col2, col3 = st.columns(3)

with col1:
    running_icon = "🟢" if status.get("running") else "🔴"
    st.metric("Estado", f"{running_icon} {'Running' if status.get('running') else 'Idle'}")

with col2:
    st.metric("Resultados", status.get("results_count", 0))

with col3:
    current = status.get("current_config", "N/A")
    st.metric("Config Actual", current if current else "N/A")

st.divider()

# ============================================================================
# MODO 1: Ver Resultados
# ============================================================================

if mode == "📊 Ver Resultados":
    st.header("📊 Resultados de Benchmarks")
    
    results = get_results()
    
    if not results:
        st.info("📭 No hay resultados aún. Ejecuta un benchmark para comenzar.")
    else:
        # Tabla de resultados
        st.subheader(f"📋 Últimos {len(results)} Resultados")
        
        df_results = pd.DataFrame(results)
        
        # Formatear timestamps
        if "timestamp" in df_results.columns:
            df_results["timestamp"] = pd.to_datetime(df_results["timestamp"])
        
        # Tabla interactiva
        st.dataframe(
            df_results[[
                "run_id", "config_name", "timestamp",
                "n_runs", "final_loss_mean"
            ]].rename(columns={
                "run_id": "Run ID",
                "config_name": "Config",
                "timestamp": "Timestamp",
                "n_runs": "Runs",
                "final_loss_mean": "Mean Loss",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        # Gráfico de comparación
        st.subheader("📈 Comparación Visual")
        
        fig = px.bar(
            df_results,
            x="config_name",
            y="final_loss_mean",
            title="Final Loss por Configuración",
            labels={"config_name": "Configuración", "final_loss_mean": "Mean Loss"},
            color="final_loss_mean",
            color_continuous_scale="RdYlGn_r",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detalles de resultado seleccionado
        st.subheader("🔍 Ver Detalles")
        
        selected_run = st.selectbox(
            "Selecciona un run",
            options=df_results["run_id"].tolist(),
            format_func=lambda x: f"{x} ({df_results[df_results['run_id']==x]['config_name'].values[0]})",
        )
        
        if selected_run and st.button("Ver Detalles Completos"):
            detail = get_result_detail(selected_run)
            
            if detail:
                st.json(detail)

# ============================================================================
# MODO 2: Ejecutar Benchmark
# ============================================================================

elif mode == "🚀 Ejecutar Benchmark":
    st.header("🚀 Ejecutar Benchmark")
    
    # Obtener configs disponibles
    available_configs = get_available_configs()
    
    if not available_configs:
        st.error("❌ No se pudieron cargar las configuraciones")
        st.stop()
    
    # Selector de config
    selected_config = st.selectbox(
        "Selecciona configuración",
        options=available_configs,
    )
    
    # Mostrar detalles de la config
    if selected_config:
        with st.expander("📋 Detalles de la Configuración"):
            config_detail = get_config_details(selected_config)
            
            if config_detail:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Reasoner**")
                    st.write(f"- Mode: `{config_detail.get('reasoner_mode')}`")
                    st.write(f"- Hidden: `{config_detail.get('n_hidden')}`")
                    st.write(f"- Blocks: `{config_detail.get('n_blocks')}`")
                    
                    st.markdown("**Training**")
                    st.write(f"- Runs: `{config_detail.get('n_runs')}`")
                    st.write(f"- Max epochs/stage: `{config_detail.get('max_epochs_per_stage')}`")
                
                with col2:
                    st.markdown("**Curriculum**")
                    st.write(f"- Enabled: `{config_detail.get('use_curriculum')}`")
                    st.write(f"- Type: `{config_detail.get('curriculum_type')}`")
                    
                    st.markdown("**Evolution**")
                    st.write(f"- Strategy: `{config_detail.get('evolution_strategy')}`")
                    st.write(f"- Mutation scale: `{config_detail.get('mutation_scale')}`")
    
    # Botón de ejecución
    st.divider()
    
    if status.get("running"):
        st.warning("⏳ Ya hay un benchmark corriendo...")
        st.info(f"Config actual: **{status.get('current_config')}**")
    else:
        if st.button("▶️  Ejecutar Benchmark", type="primary", use_container_width=True):
            if run_benchmark(selected_config):
                st.success(f"✅ Benchmark '{selected_config}' iniciado!")
                st.info("💡 Los resultados aparecerán en la sección 'Ver Resultados'")
                time.sleep(2)
                st.rerun()

# ============================================================================
# MODO 3: Comparar Configs
# ============================================================================

elif mode == "⚖️ Comparar Configs":
    st.header("⚖️ Comparación de Configuraciones")
    
    # Obtener configs disponibles
    available_configs = get_available_configs()
    
    if not available_configs:
        st.error("❌ No se pudieron cargar las configuraciones")
        st.stop()
    
    # Selector múltiple
    selected_configs = st.multiselect(
        "Selecciona configuraciones a comparar (mín. 2)",
        options=available_configs,
        default=available_configs[:2] if len(available_configs) >= 2 else available_configs,
    )
    
    # Métrica de comparación
    comparison_metric = st.selectbox(
        "Métrica principal",
        options=[
            "final_loss",
            "best_loss",
            "final_accuracy",
            "total_epochs",
            "training_time",
            "gate_diversity",
        ],
        index=0,
    )
    
    # Botón de ejecución
    st.divider()
    
    if len(selected_configs) < 2:
        st.warning("⚠️  Selecciona al menos 2 configuraciones para comparar")
    elif status.get("running"):
        st.warning("⏳ Ya hay un benchmark corriendo...")
    else:
        estimated_time = len(selected_configs) * 2  # 2 min por config (aproximado)
        
        st.info(f"⏱️ Tiempo estimado: ~{estimated_time} minutos")
        
        if st.button("▶️  Ejecutar Comparación", type="primary", use_container_width=True):
            if run_comparison(selected_configs, comparison_metric):
                st.success(f"✅ Comparación iniciada con {len(selected_configs)} configs!")
                st.info("💡 Los reportes se generarán en `data/benchmarks/reports/`")
                time.sleep(2)
                st.rerun()

# ============================================================================
# Footer & Stats
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Configs Disponibles", len(available_configs))

with col2:
    results_count = len(get_results())
    st.metric("Resultados Totales", results_count)

with col3:
    # Reportes disponibles
    reports_path = Path("data/benchmarks/reports")
    reports_count = len(list(reports_path.iterdir())) if reports_path.exists() else 0
    st.metric("Reportes", reports_count)

# Auto-refresh
if auto_refresh and status.get("running"):
    time.sleep(refresh_interval)
    st.rerun()
