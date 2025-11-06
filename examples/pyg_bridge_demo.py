"""Demostración del puente PyG conectando el grafo cognitivo híbrido."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
        from torch_geometric.utils import to_networkx  # noqa: F401  # opcional para visualizar
    except ModuleNotFoundError:
        print("[PyG Bridge Demo] Requiere torch y torch_geometric instalados.")
        return

    from core.cognitive_block import CognitiveBlock
    from core.cognitive_graph_hybrid import CognitiveGraphHybrid
    from core.latent_planner_block import LatentPlannerBlock
    from core.pyg_bridge import CognitiveGraphAdapter, GCNReasoner, GraphTrainer

    graph = CognitiveGraphHybrid()
    graph.add_block("sensor", CognitiveBlock(2, 4, 2))
    graph.add_block("planner", LatentPlannerBlock(2, 8))
    graph.add_block("decision", CognitiveBlock(8, 4, 1))
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")

    graph.forward({"sensor": [0.5, -0.2]})

    adapter = CognitiveGraphAdapter(graph)
    data = adapter.to_pyg()

    model = GCNReasoner(in_dim=data.num_features, hidden_dim=16, out_dim=1)
    trainer = GraphTrainer(model)

    target = torch.randn(data.num_nodes)

    print("=== Entrenando razonador PyG sobre CognitiveGraphHybrid ===")
    for epoch in range(10):
        loss = trainer.train_step(data, target)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

    out = trainer.infer(data)
    print("Predicciones finales:", out.view(-1).tolist())


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()
