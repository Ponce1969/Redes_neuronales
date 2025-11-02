#!/usr/bin/env python3
"""
CognitiveGraph Demo - Mente Emergente Modular
Ejemplo de una mente modular con tres subsistemas:
👁️ Percepción → 🧩 Razonamiento → 🎯 Decisión
"""

import sys
import os
import random

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.cognitive_block import CognitiveBlock
from src.core.cognitive_graph import CognitiveGraph


def main():
    print("🧠 Fase 9: CognitiveGraph - Mente Emergente Modular")
    print("=" * 60)

    # Semilla determinista para reproducibilidad de pesos aleatorios
    random.seed(42)

    # 1️⃣ Crear los bloques cognitivos
    perceptual_block = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)
    reasoning_block = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)
    decision_block = CognitiveBlock(n_inputs=1, n_hidden=2, n_outputs=1)

    # 2️⃣ Crear el grafo cognitivo
    graph = CognitiveGraph()
    graph.add_block("perception", perceptual_block)
    graph.add_block("reasoning", reasoning_block)
    graph.add_block("decision", decision_block)

    # 3️⃣ Definir conexiones entre módulos
    graph.connect("perception", "reasoning")  # percepción → razonamiento
    graph.connect("reasoning", "decision")    # razonamiento → decisión

    # 4️⃣ Mostrar resumen de la estructura
    graph.summary()

    # 5️⃣ Simular un razonamiento con datos secuenciales
    input_seq = [0.2, 0.5, 0.9, 0.4, 0.7]
    print("\n--- Ejecución paso a paso ---")
    print("Input → Percepción → Razonamiento → Decisión")
    
    for i, x in enumerate(input_seq):
        outputs = graph.step({"perception": [x]})
        perc = outputs.get("perception", [0.0])[0]
        reas = outputs.get("reasoning", [0.0])[0]
        dec = outputs.get("decision", [0.0])[0]
        
        print(f"Step {i+1}: {x:.2f} → {perc:.3f} → {reas:.3f} → {dec:.3f}")

    # 6️⃣ Mostrar estado de memoria compartida
    print("\n--- Estado de memoria compartida ---")
    memory_state = graph.get_memory_state()
    for block_name, mem in memory_state.items():
        print(f"{block_name}: {[round(m, 3) for m in mem]}")

    # 7️⃣ Demo de reseteo
    print("\n--- Reset de memoria ---")
    graph.reset_all()
    memory_after_reset = graph.get_memory_state()
    for block_name, mem in memory_after_reset.items():
        print(f"{block_name}: {[round(m, 3) for m in mem]} (reset)")

    print("\n✅ Fase 9 completada exitosamente!")
    print("🧠 Mente artificial modular operativa")


if __name__ == "__main__":
    main()
