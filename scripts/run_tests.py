#!/usr/bin/env python3
"""
Script para ejecutar todos los tests de la Fase 4B.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tests.test_gradients as test_gradients
import tests.test_stability as test_stability


def main():
    print("🧪 Neural Core - Fase 4B: Ejecutando Tests")
    print("=" * 50)
    
    try:
        print("\n1️⃣ Test de gradientes...")
        test_gradients.test_gradients_close()
        
        print("\n2️⃣ Tests de estabilidad...")
        test_stability.test_sgd_converge()
        test_stability.test_sgd_momentum_converge()
        test_stability.test_adam_converge()
        test_stability.test_rmsprop_converge()
        test_stability.test_stability_comparison()
        
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        print("\n📊 Resumen:")
        print("   ✅ Backpropagation es correcto")
        print("   ✅ Gradientes son estables")
        print("   ✅ Optimizadores convergen")
        print("   ✅ Motor neuronal está listo")
        print("\n🚀 Sistema preparado para Fase 5")
        
    except AssertionError as e:
        print(f"\n❌ Test falló: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
