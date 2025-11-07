"""
Dashboard Streamlit para Curriculum Learning System.

Monitorea y controla el entrenamiento progresivo del Reasoner.

Uso:
    PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from typing import Dict, Any, List

# Configuración de la página
st.set_page_config(
    page_title="📚 Curriculum Learning",
    page_icon="🎓",
    layout="wide",
)

# URL de la API
API_URL = "http://localhost:8000"

# ============================================================================
# Funciones Helper
# ============================================================================

def get_curriculum_status() -> Dict[str, Any]:
    """Obtiene el estado del curriculum desde la API."""
    try:
        response = requests.get(f"{API_URL}/curriculum/status", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error conectando a API: {e}")
        return {}


def start_curriculum(stages: List[Dict] = None) -> bool:
    """Inicia el curriculum."""
    try:
        payload = {"stages": stages} if stages else {}
        response = requests.post(f"{API_URL}/curriculum/start", json=payload, timeout=5)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Error iniciando curriculum: {e}")
        return False


def pause_curriculum() -> bool:
    """Pausa el curriculum."""
    try:
        response = requests.post(f"{API_URL}/curriculum/pause", timeout=2)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Error pausando: {e}")
        return False


def reset_curriculum() -> bool:
    """Resetea el curriculum."""
    try:
        response = requests.post(f"{API_URL}/curriculum/reset", timeout=2)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Error reseteando: {e}")
        return False


# ============================================================================
# Header
# ============================================================================

st.title("📚 Cognitive Curriculum Learning Dashboard")
st.markdown(
    "Sistema de aprendizaje progresivo para el Reasoner - "
    "Del simple al complejo, como un cerebro humano"
)

st.divider()

# ============================================================================
# Sidebar - Configuración
# ============================================================================

st.sidebar.header("⚙️ Configuración")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True)
refresh_interval = st.sidebar.slider(
    "Intervalo (segundos)",
    min_value=1,
    max_value=10,
    value=2,
    disabled=not auto_refresh,
)

st.sidebar.divider()

# Curriculum presets
st.sidebar.subheader("📋 Curriculum Presets")

preset_choice = st.sidebar.selectbox(
    "Selecciona preset",
    ["Estándar (7 etapas)", "Rápido (4 etapas)", "Avanzado (10 etapas)", "Personalizado"],
)

if preset_choice == "Personalizado":
    st.sidebar.info("💡 Define etapas personalizadas en el código")

st.sidebar.divider()

# Controles
st.sidebar.subheader("🎮 Controles")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start", use_container_width=True):
        if start_curriculum():
            st.success("✅ Curriculum iniciado")
            time.sleep(1)
            st.rerun()

with col2:
    if st.button("⏸️ Pause", use_container_width=True):
        if pause_curriculum():
            st.warning("⏸️  Curriculum pausado")
            time.sleep(1)
            st.rerun()

if st.sidebar.button("🔄 Reset", use_container_width=True):
    if reset_curriculum():
        st.info("🔄 Curriculum reseteado")
        time.sleep(1)
        st.rerun()

# ============================================================================
# Main Content
# ============================================================================

# Obtener estado
status = get_curriculum_status()

if not status:
    st.warning("⚠️  No se pudo obtener el estado del curriculum. ¿Está el servidor corriendo?")
    st.code(f"PYTHONPATH=src uv run uvicorn api.server:app --reload")
    st.stop()

# ============================================================================
# Estado General
# ============================================================================

st.header("📊 Estado General")

# Progress bar
if status.get('total_stages', 0) > 0:
    progress = status.get('progress', 0) / 100
    st.progress(progress, text=f"Progreso: {status['progress']:.1f}%")

# Métricas principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    running_icon = "🟢" if status.get('running') else "🔴"
    st.metric(
        "Estado",
        f"{running_icon} {'Running' if status.get('running') else 'Stopped'}"
    )

with col2:
    st.metric(
        "Etapa Actual",
        f"{status.get('current_stage_idx', 0) + 1}/{status.get('total_stages', 0)}"
    )

with col3:
    st.metric(
        "Completadas",
        f"{status.get('stages_completed', 0)}"
    )

with col4:
    current_name = status.get('current_stage_name', 'N/A')
    st.metric("Nombre", current_name if current_name else "N/A")

# Estado detallado
if status.get('running'):
    st.success(f"✅ Curriculum en ejecución - Etapa: **{current_name}**")
elif status.get('paused'):
    st.warning("⏸️  Curriculum pausado")
else:
    st.info("⏹️  Curriculum detenido")

st.divider()

# ============================================================================
# Historial de Etapas
# ============================================================================

st.header("📈 Historial de Progreso")

history = status.get('history', [])

if history:
    # Convertir a DataFrame
    df = pd.DataFrame(history)
    
    # Gráfico de pérdida por etapa
    fig_loss = go.Figure()
    
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df['mse_loss'],
        mode='lines+markers',
        name='MSE Loss',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10),
    ))
    
    fig_loss.update_layout(
        title="Evolución del Loss por Etapa",
        xaxis_title="Etapa",
        yaxis_title="MSE Loss",
        hovermode='x unified',
        height=400,
    )
    
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Gráfico de epochs por etapa
    col1, col2 = st.columns(2)
    
    with col1:
        fig_epochs = px.bar(
            df,
            x='stage',
            y='epochs',
            title="Epochs por Etapa",
            color='difficulty',
            color_continuous_scale='Viridis',
        )
        fig_epochs.update_layout(height=350)
        st.plotly_chart(fig_epochs, use_container_width=True)
    
    with col2:
        # Accuracy si está disponible
        if 'accuracy' in df.columns:
            fig_acc = px.line(
                df,
                x='stage',
                y='accuracy',
                title="Accuracy por Etapa",
                markers=True,
            )
            fig_acc.update_layout(height=350)
            fig_acc.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("📊 Accuracy no disponible para estas tareas")
    
    st.divider()
    
    # Tabla detallada
    st.subheader("📋 Tabla Detallada")
    
    # Formatear tabla
    display_df = df.copy()
    
    # Añadir íconos de estado
    display_df['Status'] = display_df.get('partial', False).apply(
        lambda x: "⚠️  Parcial" if x else "✅ Completo"
    )
    
    # Formatear columnas numéricas
    if 'mse_loss' in display_df.columns:
        display_df['MSE Loss'] = display_df['mse_loss'].apply(lambda x: f"{x:.4f}")
    
    if 'accuracy' in display_df.columns:
        display_df['Accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.1%}")
    
    if 'gate_diversity' in display_df.columns:
        display_df['Gate Diversity'] = display_df['gate_diversity'].apply(lambda x: f"{x:.3f}")
    
    # Seleccionar columnas para mostrar
    columns_to_show = ['Status', 'stage', 'difficulty', 'epochs', 'MSE Loss']
    if 'Accuracy' in display_df.columns:
        columns_to_show.append('Accuracy')
    if 'Gate Diversity' in display_df.columns:
        columns_to_show.append('Gate Diversity')
    
    st.dataframe(
        display_df[columns_to_show],
        use_container_width=True,
        hide_index=True,
    )
    
    # Estadísticas globales
    st.divider()
    st.subheader("📊 Estadísticas Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_epochs = df['epochs'].sum()
        st.metric("Total Epochs", f"{total_epochs}")
    
    with col2:
        avg_loss = df['mse_loss'].mean()
        st.metric("Avg Loss", f"{avg_loss:.4f}")
    
    with col3:
        if 'accuracy' in df.columns:
            avg_acc = df['accuracy'].mean()
            st.metric("Avg Accuracy", f"{avg_acc:.1%}")
        else:
            st.metric("Avg Accuracy", "N/A")
    
    with col4:
        completion = len(history) / status.get('total_stages', 1) * 100
        st.metric("Completion", f"{completion:.0f}%")

else:
    st.info("📭 No hay historial aún. Inicia el curriculum para comenzar.")

st.divider()

# ============================================================================
# Lista de Etapas
# ============================================================================

st.header("📚 Etapas del Curriculum")

stage_names = status.get('stage_names', [])

if stage_names:
    # Crear tabla de etapas
    stage_data = []
    
    for idx, name in enumerate(stage_names):
        completed = idx < status.get('stages_completed', 0)
        current = idx == status.get('current_stage_idx', -1)
        
        stage_data.append({
            'N°': idx + 1,
            'Estado': '🟢 Actual' if current else ('✅ Completo' if completed else '⏳ Pendiente'),
            'Nombre': name,
        })
    
    stage_df = pd.DataFrame(stage_data)
    st.dataframe(stage_df, use_container_width=True, hide_index=True)

else:
    st.info("📋 No hay etapas configuradas")

# ============================================================================
# Auto-refresh
# ============================================================================

if auto_refresh and status.get('running'):
    time.sleep(refresh_interval)
    st.rerun()
