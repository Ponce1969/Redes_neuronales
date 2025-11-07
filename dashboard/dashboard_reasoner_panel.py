"""
Dashboard de Control del Reasoner - Streamlit

Panel interactivo para gestionar el Reasoner:
- Visualización de gates en tiempo real
- Control de evolución (start/stop)
- Configuración de modos de gating
- Persistencia (save/load)
- Métricas y estado
"""

import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

API_URL = "http://localhost:8000"
REASONER_API = f"{API_URL}/reasoner"

st.set_page_config(
    page_title="🧠 Reasoner Control Panel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# FUNCIONES DE API
# ============================================================================


def get_status() -> Dict:
    """Obtiene estado del Reasoner."""
    try:
        response = requests.get(f"{REASONER_API}/status", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_recent_gates(n: int = 10) -> Dict:
    """Obtiene gates recientes."""
    try:
        response = requests.get(f"{REASONER_API}/gates?n={n}", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def start_evolution(generations: int, pop_size: int, mutation_scale: float) -> Dict:
    """Inicia evolución del Reasoner."""
    try:
        payload = {
            "generations": generations,
            "pop_size": pop_size,
            "mutation_scale": mutation_scale,
        }
        response = requests.post(f"{REASONER_API}/evolve", json=payload, timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def stop_evolution() -> Dict:
    """Detiene evolución."""
    try:
        response = requests.post(f"{REASONER_API}/evolve/stop", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def save_reasoner() -> Dict:
    """Guarda estado del Reasoner."""
    try:
        response = requests.post(f"{REASONER_API}/save", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_reasoner() -> Dict:
    """Carga estado del Reasoner."""
    try:
        response = requests.post(f"{REASONER_API}/load", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🧠 Cognitive Reasoner Control Panel")
st.markdown("Control y monitoreo del Reasoner para gating selectivo de bloques cognitivos")
st.markdown("---")

# ============================================================================
# SIDEBAR: CONTROLES
# ============================================================================

with st.sidebar:
    st.header("⚙️ Controles")
    
    st.subheader("📊 Evolución")
    
    generations = st.slider("Generaciones", 10, 200, 50, 10)
    pop_size = st.slider("Población", 4, 20, 8, 2)
    mutation_scale = st.slider("Mutación", 0.01, 0.1, 0.03, 0.01)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Evolve", use_container_width=True):
            result = start_evolution(generations, pop_size, mutation_scale)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            elif result.get("started"):
                st.success("Evolución iniciada!")
            else:
                st.warning("Ya hay evolución corriendo")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            result = stop_evolution()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.info(result.get("message", "Detenido"))
    
    st.markdown("---")
    
    st.subheader("💾 Persistencia")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("💾 Save", use_container_width=True):
            result = save_reasoner()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            elif result.get("saved"):
                st.success("Guardado!")
    
    with col4:
        if st.button("📂 Load", use_container_width=True):
            result = load_reasoner()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            elif result.get("loaded"):
                st.success("Cargado!")
    
    st.markdown("---")
    
    st.subheader("🔄 Actualización")
    refresh_rate = st.slider("Intervalo (s)", 1, 10, 2, 1)
    auto_refresh = st.checkbox("Auto-refresh", value=True)

# ============================================================================
# MÉTRICAS Y ESTADO
# ============================================================================

status_placeholder = st.empty()
metrics_placeholder = st.empty()

# ============================================================================
# VISUALIZACIÓN DE GATES
# ============================================================================

st.subheader("🎯 Gates por Bloque (Últimos)")

chart_placeholder = st.empty()
table_placeholder = st.empty()

# ============================================================================
# LOOP DE ACTUALIZACIÓN
# ============================================================================

if auto_refresh:
    iteration = 0
    
    while True:
        # Obtener estado
        status = get_status()
        
        if "error" not in status:
            # Mostrar estado
            with status_placeholder.container():
                if status.get("running"):
                    st.info(f"🔄 **Evolución en curso**: Generación {status['generation']}/{status['total_generations']} ({status['progress']:.1f}%)")
                else:
                    st.success("✅ **Reasoner listo** (no hay evolución en curso)")
            
            # Mostrar métricas
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    loss = status.get('best_loss', 1.0)
                    if loss is None:
                        loss = 1.0
                    st.metric("Best Loss", f"{loss:.4f}")
                
                with col2:
                    st.metric("Generación", f"{status.get('generation', 0)}")
                
                with col3:
                    st.metric("Predict Calls", status.get('predict_calls', 0))
                
                with col4:
                    st.metric("Evolution Runs", status.get('evolution_runs', 0))
        
        # Obtener gates recientes
        gates_data = get_recent_gates(n=10)
        
        if "error" not in gates_data and gates_data.get("gates_history"):
            gates_history = gates_data["gates_history"]
            
            # Usar el último gate para visualización
            if gates_history:
                latest_gates = gates_history[-1]
                
                # Preparar datos para gráfico
                df = pd.DataFrame([
                    {"Block": f"Block_{idx}", "Gate": float(gate)}
                    for idx, gate in latest_gates.items()
                ])
                
                # Gráfico de barras
                fig = px.bar(
                    df,
                    x="Block",
                    y="Gate",
                    color="Gate",
                    color_continuous_scale="Viridis",
                    range_y=[0, 1],
                    title="Gates Actuales por Bloque",
                    labels={"Gate": "Activación", "Block": "Bloque"},
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Bloque Cognitivo",
                    yaxis_title="Gate (0-1)",
                )
                
                with chart_placeholder:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tabla con historial
                if len(gates_history) > 1:
                    history_df = []
                    for i, gates in enumerate(gates_history[-5:]):  # Últimos 5
                        row = {"Step": len(gates_history) - 5 + i}
                        row.update({f"Block_{idx}": f"{gate:.3f}" for idx, gate in gates.items()})
                        history_df.append(row)
                    
                    with table_placeholder:
                        st.dataframe(
                            pd.DataFrame(history_df),
                            use_container_width=True,
                            hide_index=True,
                        )
        
        # Esperar antes de siguiente actualización
        iteration += 1
        time.sleep(refresh_rate)
        
        if iteration > 1000:  # Evitar overflow
            iteration = 0
else:
    st.info("Auto-refresh deshabilitado. Habilítalo en la sidebar para ver actualizaciones en tiempo real.")
    
    # Mostrar datos estáticos
    status = get_status()
    if "error" not in status:
        st.json(status)
    else:
        st.error(f"Error conectando a API: {status['error']}")
        st.info("Asegúrate que el servidor esté corriendo: `PYTHONPATH=src uv run uvicorn api.server:app --reload`")
