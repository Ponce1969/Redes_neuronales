"""
Sistema de checkpointing para Curriculum Learning.

Permite guardar y restaurar el estado completo del entrenamiento curriculum.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class CurriculumCheckpointer:
    """
    Gestiona checkpoints del curriculum learning.
    
    Características:
    - Auto-save después de cada etapa
    - Resume desde última etapa completada
    - Versionado de checkpoints
    - Backup automático
    """
    
    def __init__(self, checkpoint_dir: str = "data/curriculum"):
        """
        Inicializa el checkpointer.
        
        Args:
            checkpoint_dir: Directorio donde guardar checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "curriculum_state.json"
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def save(
        self,
        current_stage_idx: int,
        total_stages: int,
        stage_names: list,
        history: list,
        reasoner_manager,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Guarda el estado completo del curriculum.
        
        Args:
            current_stage_idx: Índice de la etapa actual
            total_stages: Total de etapas en el curriculum
            stage_names: Nombres de todas las etapas
            history: Historial de performance por etapa
            reasoner_manager: ReasonerManager para guardar su estado
            extra_info: Información adicional opcional
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "current_stage_idx": current_stage_idx,
            "total_stages": total_stages,
            "stages_completed": current_stage_idx,
            "stage_names": stage_names,
            "history": history,
            "reasoner_saved": False,
            "extra_info": extra_info or {},
        }
        
        # Backup del archivo anterior si existe
        if self.checkpoint_file.exists():
            self._create_backup()
        
        # Guardar estado del curriculum
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Guardar estado del Reasoner
        try:
            reasoner_manager.save()
            state["reasoner_saved"] = True
            # Re-guardar con flag actualizado
            with open(self.checkpoint_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️  No se pudo guardar Reasoner: {e}")
        
        print(f"💾 Checkpoint guardado: Etapa {current_stage_idx}/{total_stages}")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Carga el estado del curriculum desde el último checkpoint.
        
        Returns:
            Diccionario con el estado, o None si no existe
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            
            print(
                f"📂 Checkpoint cargado: "
                f"{state['stages_completed']}/{state['total_stages']} etapas completadas"
            )
            print(f"   Última actualización: {state['timestamp']}")
            
            return state
        
        except Exception as e:
            print(f"❌ Error cargando checkpoint: {e}")
            return None
    
    def reset(self):
        """
        Elimina el checkpoint actual (para empezar desde cero).
        """
        if self.checkpoint_file.exists():
            self._create_backup()
            self.checkpoint_file.unlink()
            print("🗑️  Checkpoint eliminado (backup creado)")
    
    def _create_backup(self):
        """Crea un backup del checkpoint actual."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"curriculum_state_{timestamp}.json"
        
        try:
            shutil.copy2(self.checkpoint_file, backup_file)
            
            # Mantener solo los últimos 5 backups
            self._cleanup_old_backups(keep=5)
        
        except Exception as e:
            print(f"⚠️  No se pudo crear backup: {e}")
    
    def _cleanup_old_backups(self, keep: int = 5):
        """
        Elimina backups antiguos, manteniendo solo los más recientes.
        
        Args:
            keep: Número de backups a mantener
        """
        backups = sorted(
            self.backup_dir.glob("curriculum_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        # Eliminar backups viejos
        for old_backup in backups[keep:]:
            try:
                old_backup.unlink()
            except Exception:
                pass
    
    def get_backup_list(self) -> list:
        """
        Lista todos los backups disponibles.
        
        Returns:
            Lista de paths a backups ordenados por fecha (más reciente primero)
        """
        backups = sorted(
            self.backup_dir.glob("curriculum_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        return [
            {
                "path": str(backup),
                "timestamp": datetime.fromtimestamp(backup.stat().st_mtime).isoformat(),
                "size_kb": backup.stat().st_size / 1024,
            }
            for backup in backups
        ]
    
    def restore_from_backup(self, backup_path: str):
        """
        Restaura un backup específico.
        
        Args:
            backup_path: Path al archivo de backup
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup no encontrado: {backup_path}")
        
        # Backup del estado actual antes de sobreescribir
        if self.checkpoint_file.exists():
            self._create_backup()
        
        shutil.copy2(backup_file, self.checkpoint_file)
        print(f"♻️  Checkpoint restaurado desde: {backup_file.name}")
