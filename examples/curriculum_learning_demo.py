"""
Demo de Curriculum Learning System.

Muestra cómo entrenar el Reasoner de manera progresiva usando
un curriculum de tareas con dificultad creciente.

Uso:
    PYTHONPATH=src python examples/curriculum_learning_demo.py
"""

import sys
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.curriculum import (
    CurriculumManager,
    CurriculumStage,
    create_standard_curriculum,
    tasks,
)
from core.reasoning.reasoner_manager import ReasonerManager
from core.cognitive_graph_hybrid import CognitiveGraphHybrid
from core.cognitive_block import CognitiveBlock


def create_simple_graph():
    """Crea un grafo cognitivo simple para el demo."""
    graph = CognitiveGraphHybrid()
    
    # Bloques básicos
    sensor = CognitiveBlock(input_dim=8, hidden_dim=16, name="sensor")
    planner = CognitiveBlock(input_dim=16, hidden_dim=16, name="planner")
    decision = CognitiveBlock(input_dim=16, hidden_dim=8, name="decision")
    
    graph.add_block("sensor", sensor)
    graph.add_block("planner", planner)
    graph.add_block("decision", decision)
    
    graph.connect("sensor", "planner")
    graph.connect("planner", "decision")
    
    return graph


def main():
    """Ejecuta el demo de curriculum learning."""
    print("="*70)
    print("🎓 CURRICULUM LEARNING SYSTEM - DEMO")
    print("="*70)
    print("\nEste demo entrena al Reasoner usando un curriculum progresivo:")
    print("1. Identity (trivial)")
    print("2. XOR (no lineal simple)")
    print("3. Parity-3 (complejidad media)")
    print("4. Counting (agregación)")
    print("\n" + "="*70 + "\n")
    
    # 1. Crear grafo cognitivo
    print("📊 Creando grafo cognitivo...")
    graph = create_simple_graph()
    
    # 2. Crear ReasonerManager
    print("🧠 Inicializando Reasoner...")
    reasoner_manager = ReasonerManager(
        n_inputs=24,  # Depende del grafo
        n_hidden=48,
        n_blocks=3,   # sensor, planner, decision
    )
    
    # 3. Crear CurriculumManager
    print("📚 Configurando curriculum...\n")
    manager = CurriculumManager(
        reasoner_manager=reasoner_manager,
        graph=graph,
        auto_save=True,
    )
    
    # 4. Añadir etapas personalizadas (más rápidas para demo)
    stages = [
        CurriculumStage(
            name="identity",
            task_generator=lambda: tasks.identity_task(n_features=2, samples=16),
            difficulty=1,
            max_epochs=20,
            success_threshold=0.02,
            fail_threshold=0.15,
            log_interval=5,
        ),
        CurriculumStage(
            name="xor",
            task_generator=lambda: tasks.xor_task(samples=16),
            difficulty=2,
            max_epochs=30,
            success_threshold=0.03,
            fail_threshold=0.2,
            log_interval=5,
        ),
        CurriculumStage(
            name="parity-3",
            task_generator=lambda: tasks.parity_task(n_bits=3, samples=24),
            difficulty=3,
            max_epochs=40,
            success_threshold=0.04,
            fail_threshold=0.25,
            log_interval=10,
        ),
        CurriculumStage(
            name="counting",
            task_generator=lambda: tasks.counting_task(max_value=4, samples=24),
            difficulty=4,
            max_epochs=50,
            success_threshold=0.05,
            fail_threshold=0.3,
            log_interval=10,
        ),
    ]
    
    for stage in stages:
        manager.add_stage(stage)
    
    # 5. Ejecutar curriculum
    print("🚀 Iniciando entrenamiento curriculum...\n")
    
    try:
        history = manager.run()
        
        # 6. Mostrar resultados
        print("\n" + "="*70)
        print("✅ CURRICULUM COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        if history:
            print("\n📈 Progresión del aprendizaje:")
            for i, record in enumerate(history, 1):
                status_icon = "✅" if not record.get('partial', False) else "⚠️ "
                print(
                    f"{status_icon} Etapa {i}: {record['stage']:12s} | "
                    f"Epochs: {record['epochs']:3d} | "
                    f"Loss: {record['mse_loss']:.4f} | "
                    f"Acc: {record.get('accuracy', 0):.1%}"
                )
            
            print(f"\n🏆 Total epochs: {sum(r['epochs'] for r in history)}")
            print(f"🎯 Promedio loss final: {sum(r['mse_loss'] for r in history) / len(history):.4f}")
        
        print("\n💾 Estado del Reasoner guardado automáticamente")
        print("   Ubicación: data/persistence/reasoner_state.npz")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por usuario")
        print("   Progreso guardado en checkpoint")
        status = manager.status()
        print(f"   Etapas completadas: {status['stages_completed']}/{status['total_stages']}")
    
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Demo finalizado")
    print("="*70)


if __name__ == "__main__":
    main()
