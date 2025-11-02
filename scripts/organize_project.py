#!/usr/bin/env python3
"""
Script para organizar y limpiar el proyecto en Linux.
Usa python3 y uv para ejecutar comandos.
"""

import os
import subprocess

# Directorios base
BASE_DIR = "/home/gonzapython/Documentos/Redes_Neuronales/neural_core"

# Archivos a organizar
FILES_TO_ORGANIZE = {
    "debug_fase5.py": "scripts/",
    "fase5_funcional.py": "scripts/",
    "test_fase5_final.py": "tests/",
    "run_tests.py": "scripts/",
}

# Crear directorios si no existen
os.makedirs(os.path.join(BASE_DIR, "scripts"), exist_ok=True)

print("🧹 Organizando proyecto...")
print("=" * 50)

for filename, dest_dir in FILES_TO_ORGANIZE.items():
    src_path = os.path.join(BASE_DIR, filename)
    dest_path = os.path.join(BASE_DIR, dest_dir, filename)
    
    if os.path.exists(src_path):
        # Usar mv para mover archivos en Linux
        dest_full_path = os.path.join(BASE_DIR, dest_dir, filename)
        os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)
        
        # Mover archivo con mv
        result = subprocess.run(['mv', src_path, dest_full_path], 
                              capture_output=True, text=True, cwd=BASE_DIR)
        
        if result.returncode == 0:
            print(f"   ✅ {filename} -> {dest_dir}{filename}")
        else:
            print(f"   ❌ Error moviendo {filename}: {result.stderr}")
    else:
        print(f"   ⚠️ {filename} no encontrado")

# Limpiar archivos temporales
TEMP_FILES = [
    "debug_fase5.py",
    "fase5_funcional.py",
    "test_fase5_final.py",
]

print("\n🗑️ Limpieza de archivos temporales...")
for filename in TEMP_FILES:
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"   🧹 Eliminado: {filename}")

print("\n📋 Estructura final:")
print("   src/           # Código fuente principal")
print("   tests/         # Tests de validación")
print("   examples/      # Ejemplos prácticos")
print("   scripts/       # Scripts de diagnóstico y utilidades")
print("   docs/          # Documentación")
print("   ✅ Proyecto organizado exitosamente")

print("\n🐧 Comandos para Linux:")
print("   python3 organize_project.py")
print("   uv run python3 tests/test_latent_integration.py")
print("   uv run python3 examples/pretrain_with_z.py")
