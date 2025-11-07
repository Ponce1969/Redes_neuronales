"""
Demo: Cognitive Reasoning Layer - Control selectivo de bloques mediante Reasoner.

Este ejemplo muestra cómo el Reasoner decide qué bloques activar en el grafo cognitivo,
permitiendo rutas cognitivas dinámicas basadas en los planes latentes.
"""

import numpy as np

from core.cognitive_block import CognitiveBlock
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.latent_planner_block import LatentPlannerBlock
from core.reasoning.reasoner import Reasoner


def demo() -> None:
    """Demostración básica del Reasoner con grafo híbrido."""
    print("=" * 70)
    print("🧠 Cognitive Reasoning Layer Demo - Fase 31-B")
    print("=" * 70)

    # 1) Construir grafo cognitivo con diferentes tipos de bloques
    print("\n[1/4] Construyendo grafo cognitivo híbrido...")
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

    print(f"   ✓ Bloques creados: {list(graph.blocks.keys())}")
    print(f"   ✓ Conexiones establecidas")

    # 2) Crear Reasoner
    # Dimensión de entrada: estimamos tamaño conservador para concatenación
    per_block_dim = 8  # Reasoner hará pad/truncate automáticamente
    n_blocks = len(graph.blocks)
    print(f"\n[2/4] Creando Reasoner (n_blocks={n_blocks}, hidden=32)...")

    reasoner = Reasoner(
        n_inputs=per_block_dim * n_blocks, n_hidden=32, n_blocks=n_blocks, seed=42
    )
    print("   ✓ Reasoner inicializado")

    # 3) Ejecutar inferencias de prueba con diferentes modos
    print("\n[3/4] Ejecutando inferencias con diferentes modos de gating...")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    modes = [
        ("softmax", {"mode": "softmax", "temp": 1.0}),
        ("top-2", {"mode": "topk", "top_k": 2}),
        ("threshold", {"mode": "threshold"}),
    ]

    for mode_name, mode_params in modes:
        print(f"\n   📊 Modo: {mode_name}")
        print("   " + "-" * 60)

        for idx, x in enumerate(X):
            # Forward normal primero para que planner compute last_plan
            _ = graph.forward({"sensor": x.tolist()})

            # Forward con reasoner
            out = graph.forward_with_reasoner({"sensor": x.tolist()}, reasoner, **mode_params)
            y = list(out.values())[-1].data.squeeze()

            # Obtener gates aplicados
            gates = graph.last_gates if hasattr(graph, "last_gates") else {}  # type: ignore[attr-defined]
            gates_str = ", ".join(
                [f"{name}={gates.get(name, 0):.3f}" for name in graph.blocks.keys()]
            )

            print(f"   Input {idx}: {x.tolist()} -> Output: {float(y):.4f}")
            print(f"           Gates: [{gates_str}]")

    # 4) Comparar forward normal vs con reasoner
    print("\n[4/4] Comparación Forward Normal vs Reasoner...")
    print("   " + "-" * 60)

    test_x = np.array([0.5, 0.5], dtype=np.float32)

    # Forward normal
    out_normal = graph.forward({"sensor": test_x.tolist()})
    y_normal = list(out_normal.values())[-1].data.squeeze()

    # Forward con reasoner (softmax)
    out_reasoner = graph.forward_with_reasoner(
        {"sensor": test_x.tolist()}, reasoner, mode="softmax"
    )
    y_reasoner = list(out_reasoner.values())[-1].data.squeeze()

    print(f"   Input: {test_x.tolist()}")
    print(f"   Normal output:   {float(y_normal):.6f}")
    print(f"   Reasoner output: {float(y_reasoner):.6f}")
    print(f"   Diferencia:      {abs(float(y_normal) - float(y_reasoner)):.6f}")

    print("\n" + "=" * 70)
    print("✅ Demo completada - El Reasoner controla selectivamente los bloques")
    print("=" * 70)
    print("\n💡 Próximos pasos:")
    print("   1. Entrena el Reasoner con `evolve_reasoner_on_task()` (ver training helper)")
    print("   2. Visualiza gates en el dashboard PyG con colores dinámicos")
    print("   3. Integra con monitor para tracking de decisiones")
    print("=" * 70)


if __name__ == "__main__":
    demo()
