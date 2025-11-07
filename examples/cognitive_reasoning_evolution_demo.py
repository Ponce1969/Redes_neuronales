"""
Demo: Entrenamiento evolutivo del Reasoner sobre XOR.

Este ejemplo muestra cómo evolucionar el Reasoner para mejorar el desempeño
del grafo cognitivo en una tarea específica (XOR), sin modificar los pesos
de los bloques del grafo.
"""

import numpy as np

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.latent_planner_block import LatentPlannerBlock
from core.reasoning import (
    Reasoner,
    evaluate_reasoner,
    evolve_reasoner_on_task,
    extract_gates_history,
)


def build_graph() -> CognitiveGraphHybrid:
    """Construye un grafo cognitivo híbrido de prueba."""
    graph = CognitiveGraphHybrid()

    graph.add_block("sensor", CognitiveBlock(n_inputs=2, n_hidden=4, n_outputs=2))
    graph.add_block(
        "planner", LatentPlannerBlock(n_in=2, n_hidden=8, max_steps=4, retain_plan=True)
    )
    graph.add_block("memory", CognitiveBlock(n_inputs=8, n_hidden=6, n_outputs=4))
    graph.add_block("decision", CognitiveBlock(n_inputs=4, n_hidden=3, n_outputs=1))

    graph.connect("sensor", "planner")
    graph.connect("planner", "memory")
    graph.connect("memory", "decision")

    return graph


def demo() -> None:
    """Demo completo de entrenamiento evolutivo del Reasoner."""
    print("=" * 80)
    print("🧬 Cognitive Reasoning Evolution Demo - Fase 31-B")
    print("=" * 80)

    # 1) Preparar dataset XOR
    print("\n[1/5] Preparando dataset XOR...")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([0, 1, 1, 0], dtype=np.float32)
    print(f"   ✓ Dataset: {len(X)} ejemplos")

    # 2) Construir grafo
    print("\n[2/5] Construyendo grafo cognitivo...")
    graph = build_graph()
    print(f"   ✓ Bloques: {list(graph.blocks.keys())}")

    # 3) Crear Reasoner inicial
    print("\n[3/5] Inicializando Reasoner...")
    per_block_dim = 8
    n_blocks = len(graph.blocks)

    reasoner_initial = Reasoner(
        n_inputs=per_block_dim * n_blocks, n_hidden=32, n_blocks=n_blocks, seed=42
    )

    # Evaluar loss inicial
    loss_initial = evaluate_reasoner(graph, reasoner_initial, X, Y, mode="softmax")
    print(f"   ✓ Loss inicial: {loss_initial:.6f}")

    # 4) Evolucionar Reasoner
    print("\n[4/5] Evolucionando Reasoner (50 generaciones, población=10)...")
    print("-" * 80)

    reasoner_evolved, loss_history = evolve_reasoner_on_task(
        graph=graph,
        base_reasoner=reasoner_initial,
        X=X,
        Y=Y,
        generations=50,
        pop_size=10,
        mutation_scale=0.03,
        mode="softmax",
        verbose=True,
    )

    print("-" * 80)

    # 5) Analizar resultados
    print("\n[5/5] Análisis de resultados...")
    print("-" * 80)

    # Comparar predicciones antes y después
    print("\n   📊 Comparación de predicciones:")
    print("   " + "-" * 60)
    print(f"   {'Input':<12} {'Target':<10} {'Inicial':<12} {'Evolucionado':<12} {'Mejora'}")
    print("   " + "-" * 60)

    for idx, (x, y_true) in enumerate(zip(X, Y)):
        # Predicción con reasoner inicial
        _ = graph.forward({"sensor": x.tolist()})
        out_init = graph.forward_with_reasoner(
            {"sensor": x.tolist()}, reasoner_initial, mode="softmax"
        )
        y_init = float(list(out_init.values())[-1].data.squeeze())

        # Predicción con reasoner evolucionado
        _ = graph.forward({"sensor": x.tolist()})
        out_evol = graph.forward_with_reasoner(
            {"sensor": x.tolist()}, reasoner_evolved, mode="softmax"
        )
        y_evol = float(list(out_evol.values())[-1].data.squeeze())

        # Calcular errores
        err_init = abs(y_init - y_true)
        err_evol = abs(y_evol - y_true)
        improvement = "✓" if err_evol < err_init else "→"

        print(
            f"   {str(x.tolist()):<12} {y_true:<10.1f} {y_init:<12.4f} "
            f"{y_evol:<12.4f} {improvement}"
        )

    # Analizar gates aplicados
    print("\n   🎯 Gates aplicados por el Reasoner evolucionado:")
    print("   " + "-" * 60)

    gates_history = extract_gates_history(graph, reasoner_evolved, X, mode="softmax")
    block_names = list(graph.blocks.keys())

    for idx, (x, gates) in enumerate(zip(X, gates_history)):
        gates_str = " | ".join([f"{name}: {gates.get(name, 0):.3f}" for name in block_names])
        print(f"   Input {idx} {x.tolist()}: {gates_str}")

    # Resumen estadístico
    print("\n   📈 Resumen de evolución:")
    print("   " + "-" * 60)
    print(f"   Loss inicial:      {loss_history[0]:.6f}")
    print(f"   Loss final:        {loss_history[-1]:.6f}")
    print(f"   Mejora absoluta:   {loss_history[0] - loss_history[-1]:.6f}")
    print(f"   Mejora relativa:   {((loss_history[0] - loss_history[-1]) / loss_history[0]) * 100:.2f}%")
    print(f"   Generaciones:      {len(loss_history) - 1}")

    # Graficar evolución (ASCII art básico)
    print("\n   📉 Curva de aprendizaje (cada 10 generaciones):")
    print("   " + "-" * 60)
    max_loss = max(loss_history)
    min_loss = min(loss_history)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    for gen in range(0, len(loss_history), 10):
        loss = loss_history[gen]
        normalized = (loss - min_loss) / loss_range
        bar_len = int(normalized * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"   Gen {gen:3d}: {bar} {loss:.6f}")

    print("\n" + "=" * 80)
    print("✅ Demo completada - El Reasoner ha mejorado mediante evolución")
    print("=" * 80)
    print("\n💡 Próximos pasos:")
    print("   1. Guarda el Reasoner con reasoner.state_dict() para persistencia")
    print("   2. Integra con el dashboard para visualización en tiempo real")
    print("   3. Experimenta con diferentes arquitecturas de grafo")
    print("   4. Prueba con datasets más complejos (clasificación, regresión)")
    print("=" * 80)


if __name__ == "__main__":
    demo()
