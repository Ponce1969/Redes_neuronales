"""
Demo Agentivo - MVP Día 1.

Demuestra el loop Plan-Act-Reflect funcionando con el sistema agentivo.

Uso:
    PYTHONPATH=src python examples/agentic_demo.py
"""

import sys
import asyncio
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock
from core.reasoning.reasoner_manager import ReasonerManager
from core.agents import create_default_orchestrator
from core.tools import create_default_registry


def create_demo_graph():
    """Crea un grafo cognitivo simple para demo."""
    graph = CognitiveGraphHybrid()
    
    # Bloques básicos (n_inputs, n_hidden, n_outputs)
    sensor = CognitiveBlock(n_inputs=8, n_hidden=16, n_outputs=16)
    planner = CognitiveBlock(n_inputs=16, n_hidden=16, n_outputs=16)
    decision = CognitiveBlock(n_inputs=16, n_hidden=8, n_outputs=8)
    
    graph.add_block("sensor", sensor)
    graph.add_block("planner", planner)
    graph.add_block("decision", decision)
    
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")
    
    return graph


async def main():
    """Ejecuta el demo agentivo."""
    print("\n" + "="*70)
    print("🤖 COGNITIVE AGENTIC REASONER - DEMO MVP")
    print("="*70)
    print()
    print("Este demo demuestra el loop agentivo Plan-Act-Reflect:")
    print("  1. CONTEXT:  Recopila información del sistema")
    print("  2. PLAN:     Genera plan de acciones")
    print("  3. ACT:      Ejecuta acciones con tools")
    print("  4. VERIFY:   Verifica calidad de resultados")
    print("  5. REFLECT:  Reflexiona y aprende")
    print()
    print("="*70)
    print()
    
    # 1. Setup del sistema
    print("📊 Configurando sistema cognitivo...")
    
    graph = create_demo_graph()
    print(f"   ✅ Grafo creado: {len(graph.blocks)} bloques")
    
    reasoner_manager = ReasonerManager(
        n_inputs=24,
        n_hidden=48,
        n_blocks=3,
    )
    print(f"   ✅ Reasoner inicializado: {reasoner_manager.reasoner.n_blocks} bloques, hidden={reasoner_manager.reasoner.n_hidden}")
    
    # 2. Crear tool registry
    print("\n🔧 Configurando tool system...")
    tool_registry = create_default_registry(graph, reasoner_manager)
    print(f"   ✅ Tools registrados: {tool_registry.list_tools()}")
    
    # 3. Crear orchestrator
    print("\n🧠 Configurando orchestrator agentivo...")
    orchestrator = create_default_orchestrator(
        graph=graph,
        reasoner_manager=reasoner_manager,
        goal="optimize_performance",
        verbose=True,
    )
    
    # Conectar tool registry al action agent
    orchestrator.action_agent.tool_registry = tool_registry
    
    print("   ✅ Orchestrator configurado con 5 agentes:")
    print("      - ContextAgent:   Recopila información")
    print("      - PlannerAgent:   Genera planes")
    print("      - ActionAgent:    Ejecuta acciones")
    print("      - VerifierAgent:  Verifica calidad")
    print("      - ReflectorAgent: Reflexiona y aprende")
    
    # 4. Ejecutar loop
    print("\n" + "="*70)
    print("🚀 EJECUTANDO AGENTIC LOOP")
    print("="*70)
    print()
    
    result = await orchestrator.loop(
        max_iterations=3,
        goal="optimize_performance",
        early_stop=True,
    )
    
    # 5. Resumen de resultados
    print("\n" + "="*70)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*70)
    print()
    
    print(f"✅ Loop completado exitosamente: {result['success']}")
    print(f"📈 Ciclos ejecutados: {result['iterations_run']}")
    final_decision = result.get('final_decision', 'unknown')
    if final_decision:
        print(f"🎯 Decisión final: {final_decision.upper()}")
    else:
        print(f"🎯 Decisión final: NO DECISION (aborted all cycles)")
    print(f"⏱️  Tiempo total: {result['total_time']:.2f}s")
    print()
    
    # Estadísticas de agentes
    print("📊 Estadísticas de Agentes:")
    stats = orchestrator.get_stats()
    
    for agent_name, agent_stats in stats["agents"].items():
        print(f"\n   {agent_name}:")
        print(f"      Calls: {agent_stats['call_count']}")
        print(f"      Success Rate: {agent_stats['success_rate']:.1%}")
        print(f"      Avg Time: {agent_stats['avg_time']:.3f}s")
    
    # Historial de ciclos
    print("\n" + "="*70)
    print("📜 HISTORIAL DE CICLOS")
    print("="*70)
    print()
    
    for i, cycle in enumerate(result['history'], 1):
        verification = cycle.get("verification", {})
        decision = verification.get("decision", "unknown") if verification else "aborted"
        score = verification.get("score", 0.0) if verification else 0.0
        cycle_time = cycle.get("cycle_time", 0.0)
        
        # Número de acciones ejecutadas
        observations = cycle.get("action", {}).get("observations", []) if cycle.get("action") else []
        n_actions = len(observations)
        
        # Insights de reflexión
        insights = cycle.get("reflection", {}).get("insights", []) if cycle.get("reflection") else []
        n_insights = len(insights)
        
        symbol = "✅" if decision == "accept" else ("⚠️" if decision == "retry" else "❌")
        
        print(f"Ciclo {i} {symbol}:")
        print(f"   Decisión: {decision.upper()}")
        print(f"   Score: {score:.2f}")
        print(f"   Acciones: {n_actions}")
        print(f"   Insights: {n_insights}")
        print(f"   Tiempo: {cycle_time:.2f}s")
        
        # Mostrar primer insight si existe
        if insights:
            first_insight = insights[0]
            print(f"   💡 Insight: {first_insight.get('content', 'N/A')}")
        
        print()
    
    # Tool statistics
    print("="*70)
    print("🔧 ESTADÍSTICAS DE TOOLS")
    print("="*70)
    print()
    
    tool_stats = tool_registry.get_all_stats()
    for tool_name, stats in tool_stats.items():
        if stats["call_count"] > 0:
            print(f"{tool_name}:")
            print(f"   Calls: {stats['call_count']}")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print(f"   Avg Time: {stats['avg_time']:.3f}s")
            print()
    
    # Conclusión
    print("="*70)
    print("🎉 DEMO COMPLETADO")
    print("="*70)
    print()
    print("Próximos pasos:")
    print("  1. Ver dashboard: PYTHONPATH=src streamlit run dashboard/dashboard_agentic.py")
    print("  2. Probar con API: curl http://localhost:8000/agents/run-loop")
    print("  3. Día 2: Integrar LLM-as-Judge (Gemini/DeepSeek)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
