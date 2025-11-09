"""
Dashboard Hub - Centro de Control Unificado.

Integra todos los dashboards del proyecto en una sola aplicación con pestañas.

Uso:
    PYTHONPATH=src streamlit run dashboard/dashboard_hub.py
"""

import sys
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st


def main():
    """Dashboard principal con todas las fases integradas."""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Neural Core - Control Hub",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Header principal
    st.markdown("""
    # 🧠 Neural Core - Control Hub
    ### Centro de Control Unificado del Sistema Cognitivo
    """)
    
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.markdown("## 🎯 Navegación")
        st.markdown("""
        Selecciona una pestaña para acceder a:
        
        - **🤖 Agentic**: Loop Plan-Act-Reflect (Fase 35)
        - **📊 Benchmark**: Evaluaciones científicas (Fase 34)
        - **📚 Curriculum**: Aprendizaje progresivo (Fase 33)
        - **🧠 Reasoner**: Control del razonador (Fase 32)
        - **🔴 Live**: Visualización en tiempo real
        - **📈 PyG Viz**: Grafos con PyTorch Geometric
        
        ---
        
        ### 📖 Guía Rápida
        
        **Para empezar**:
        1. Inicia el servidor API
        2. Selecciona una pestaña
        3. Interactúa con los controles
        
        **Servidor API**:
        ```bash
        PYTHONPATH=src uv run uvicorn api.server:app --reload
        ```
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Estado del Sistema")
        
        # Estado del servidor API
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                st.success("✅ API Server: Online")
            else:
                st.warning("⚠️ API Server: Error")
        except:
            st.error("❌ API Server: Offline")
        
        st.markdown("---")
        st.caption("Neural Core v1.0 | Fase 35 MVP")
    
    # Tabs principales
    tabs = st.tabs([
        "🤖 Agentic Loop",
        "📊 Benchmark Suite",
        "📚 Curriculum Learning",
        "🧠 Reasoner Control",
        "🔴 Live Stream",
        "📈 PyG Visualization",
    ])
    
    # ========================================================================
    # TAB 1: AGENTIC LOOP (FASE 35)
    # ========================================================================
    with tabs[0]:
        st.header("🤖 Agentic Reasoner Loop")
        st.markdown("**Fase 35**: Sistema agentivo Plan-Act-Reflect")
        
        st.info("""
        **Estado**: ⚠️ Dashboard en desarrollo (Día 2)
        
        Por ahora puedes:
        - Ejecutar el demo: `PYTHONPATH=src python examples/agentic_demo.py`
        - Ver la documentación: `docs/fase35_agentic_reasoners.md`
        
        **Próximamente** (Día 2):
        - Visualización del loop en tiempo real
        - Control interactivo de agentes
        - Métricas de performance
        - Historial de ciclos
        """)
        
        # Placeholder para el futuro
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estado", "En desarrollo", "Día 2")
        
        with col2:
            st.metric("Agentes", "5", "Context, Plan, Act, Verify, Reflect")
        
        with col3:
            st.metric("Tools", "5", "reasoner_evolve, graph_analyze, etc.")
        
        st.markdown("---")
        
        # Demo manual
        st.subheader("🚀 Ejecutar Demo")
        
        goal = st.selectbox(
            "Objetivo",
            ["optimize_performance", "explore", "learn", "diagnose"],
            help="Objetivo del loop agentivo"
        )
        
        max_iter = st.slider("Iteraciones máximas", 1, 10, 3)
        early_stop = st.checkbox("Early stopping", value=True)
        
        if st.button("▶️ Ejecutar Loop", type="primary"):
            st.info("⚠️ API endpoint `/agents/run-loop` pendiente (Día 2)")
            st.code(f"""
# Comando para ejecutar manualmente:
PYTHONPATH=src python examples/agentic_demo.py

# Con objetivo personalizado:
# (modificar en el código: goal="{goal}", max_iterations={max_iter})
            """, language="bash")
    
    # ========================================================================
    # TAB 2: BENCHMARK SUITE (FASE 34)
    # ========================================================================
    with tabs[1]:
        st.header("📊 Benchmark Suite Científico")
        st.markdown("**Fase 34**: Evaluaciones reproducibles con análisis estadístico")
        
        st.info("""
        Para el dashboard completo de Benchmark, ejecuta:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py
        ```
        """)
        
        # Vista resumida
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Configuraciones Disponibles")
            configs = [
                "baseline_random",
                "curriculum_softmax",
                "curriculum_topk",
                "curriculum_fast",
                "no_curriculum_topk",
                "high_mutation",
                "large_reasoner",
            ]
            for config in configs:
                st.markdown(f"- `{config}`")
        
        with col2:
            st.markdown("### 📊 Métricas")
            metrics = [
                "Final Loss",
                "Convergence Rate",
                "Stability",
                "Gate Diversity",
                "Efficiency",
            ]
            for metric in metrics:
                st.markdown(f"- {metric}")
        
        if st.button("🔗 Abrir Dashboard Completo", key="benchmark"):
            st.info("Ejecuta: `PYTHONPATH=src streamlit run dashboard/dashboard_benchmark.py`")
    
    # ========================================================================
    # TAB 3: CURRICULUM LEARNING (FASE 33)
    # ========================================================================
    with tabs[2]:
        st.header("📚 Curriculum Learning System")
        st.markdown("**Fase 33**: Entrenamiento progresivo del Reasoner")
        
        st.info("""
        Para el dashboard completo de Curriculum, ejecuta:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py
        ```
        """)
        
        # Vista resumida
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📖 Tareas del Curriculum")
            tasks = [
                "1. Identity (básica)",
                "2. XOR (lógica)",
                "3. Parity (complejidad media)",
                "4. Counting (secuencial)",
                "5. Sequence (memoria)",
                "6. Memory (largo plazo)",
                "7. Reasoning (avanzada)",
            ]
            for task in tasks:
                st.markdown(f"- {task}")
        
        with col2:
            st.markdown("### 📈 Métricas")
            st.markdown("- MSE Loss")
            st.markdown("- Accuracy")
            st.markdown("- Gate Diversity")
            st.markdown("- Gate Entropy")
            st.markdown("- Convergence Rate")
            st.markdown("- Stability")
        
        if st.button("🔗 Abrir Dashboard Completo", key="curriculum"):
            st.info("Ejecuta: `PYTHONPATH=src streamlit run dashboard/dashboard_curriculum.py`")
    
    # ========================================================================
    # TAB 4: REASONER CONTROL (FASE 32)
    # ========================================================================
    with tabs[3]:
        st.header("🧠 Reasoner Control Panel")
        st.markdown("**Fase 32**: Control y monitoreo del Reasoner")
        
        st.info("""
        Para el dashboard completo de Reasoner, ejecuta:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_reasoner_panel.py
        ```
        
        O con visualización PyG:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_pyg_with_reasoner.py
        ```
        """)
        
        # Vista resumida
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Controles")
            st.markdown("- Predicción de gates")
            st.markdown("- Evolución del reasoner")
            st.markdown("- Modos: softmax, topk, threshold")
            st.markdown("- Guardar/Cargar estado")
        
        with col2:
            st.markdown("### 📊 Visualización")
            st.markdown("- Distribución de gates")
            st.markdown("- Historial de evolución")
            st.markdown("- Grafo cognitivo interactivo")
            st.markdown("- Métricas en tiempo real")
        
        if st.button("🔗 Abrir Dashboard Completo", key="reasoner"):
            st.info("Ejecuta: `PYTHONPATH=src streamlit run dashboard/dashboard_reasoner_panel.py`")
    
    # ========================================================================
    # TAB 5: LIVE STREAM
    # ========================================================================
    with tabs[4]:
        st.header("🔴 Live Stream Visualization")
        st.markdown("Visualización en tiempo real del sistema")
        
        st.info("""
        Para el dashboard de Live Stream, ejecuta:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_live_stream.py
        ```
        """)
        
        st.markdown("### 📡 Features")
        st.markdown("- Actualización automática cada 2 segundos")
        st.markdown("- Métricas en tiempo real")
        st.markdown("- Gráficos animados")
        st.markdown("- Historial de estados")
        
        if st.button("🔗 Abrir Dashboard Completo", key="live"):
            st.info("Ejecuta: `PYTHONPATH=src streamlit run dashboard/dashboard_live_stream.py`")
    
    # ========================================================================
    # TAB 6: PyG VISUALIZATION
    # ========================================================================
    with tabs[5]:
        st.header("📈 PyTorch Geometric Visualization")
        st.markdown("Visualización avanzada con PyG")
        
        st.info("""
        Para los dashboards de visualización PyG:
        
        **Interactivo**:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_pyg_interactive.py
        ```
        
        **Con Reasoner**:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_pyg_with_reasoner.py
        ```
        
        **Básico**:
        ```bash
        PYTHONPATH=src streamlit run dashboard/dashboard_pyg_viz.py
        ```
        """)
        
        st.markdown("### 🎨 Features")
        st.markdown("- Grafos interactivos 3D")
        st.markdown("- Visualización de gates")
        st.markdown("- Análisis de conectividad")
        st.markdown("- Exportación de layouts")
        
        if st.button("🔗 Abrir Dashboard Completo", key="pyg"):
            st.info("Ejecuta uno de los comandos de arriba según tu necesidad")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🧠 <strong>Neural Core Control Hub</strong> | Fase 35 MVP Día 1 Completado</p>
        <p>Próximo: Día 2 - LLM Integration, API REST Completa, Dashboard Agentic</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
