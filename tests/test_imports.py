#!/usr/bin/env python3
"""
Test de validación de imports - Fase 7
Verifica que todos los módulos se puedan importar correctamente
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test de importación de todos los módulos"""
    print("🔍 Validando imports...")
    
    try:
        # Test autograd
        from autograd.value import Value
        print("   ✅ autograd.value importado")
        
        from autograd.ops import relu
        print("   ✅ autograd.ops importado")
        
        from autograd.functional import linear, mse_loss
        print("   ✅ autograd.functional importado")
        
        # Test core
        from core.memory_cell import MemoryCell
        print("   ✅ core.memory_cell importado")
        
        from core.macro_neuron import MacroNeuron
        print("   ✅ core.macro_neuron importado")
        
        # Test funcional
        v1 = Value(1.0)
        v2 = Value(2.0)
        result = v1 + v2
        print("   ✅ Operaciones Value funcionando")
        
        memory = MemoryCell(size=2)
        print("   ✅ MemoryCell creada")
        
        macro = MacroNeuron(n_inputs=1, n_hidden=2)
        print("   ✅ MacroNeuron creada")
        
        print("\n🎉 ¡Todos los imports funcionan correctamente!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
