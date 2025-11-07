"""
Sistema de Provenance - Rastreo completo de reproducibilidad.

Captura toda la información necesaria para reproducir exactamente un experimento.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import platform
import sys
import json
import subprocess
import numpy as np


@dataclass
class BenchmarkProvenance:
    """
    Provenance completa de un experimento.
    
    Características:
    - ✅ Environment capture (Python, NumPy, OS)
    - ✅ Git state (commit, branch, dirty)
    - ✅ Random seeds
    - ✅ Timestamp
    - ✅ Config hash
    """
    
    # ========================================================================
    # Identidad del Run
    # ========================================================================
    run_id: str
    timestamp: str
    config_hash: str
    config_name: str
    
    # ========================================================================
    # Environment
    # ========================================================================
    python_version: str
    numpy_version: str
    system_os: str
    system_platform: str
    system_machine: str
    cpu_count: int
    
    # ========================================================================
    # Code Version
    # ========================================================================
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_is_dirty: bool = False
    git_remote_url: Optional[str] = None
    
    # ========================================================================
    # Random State
    # ========================================================================
    random_seed: int = 42
    numpy_random_state: Optional[str] = None  # Serializado
    
    # ========================================================================
    # Full Config
    # ========================================================================
    full_config: Dict[str, Any] = None
    
    @classmethod
    def capture(
        cls,
        config: 'BenchmarkConfig',
        run_id: Optional[str] = None,
    ) -> 'BenchmarkProvenance':
        """
        Captura provenance del experimento actual.
        
        Args:
            config: BenchmarkConfig del experimento
            run_id: ID único del run (auto-generado si None)
        
        Returns:
            BenchmarkProvenance completo
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Git info
        git_commit = cls._get_git_commit()
        git_branch = cls._get_git_branch()
        git_is_dirty = cls._is_git_dirty()
        git_remote_url = cls._get_git_remote_url()
        
        # NumPy random state
        numpy_state = np.random.get_state()
        numpy_state_serialized = cls._serialize_numpy_state(numpy_state)
        
        return cls(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config_hash=config.hash(),
            config_name=config.name,
            # Environment
            python_version=sys.version,
            numpy_version=np.__version__,
            system_os=platform.system(),
            system_platform=platform.platform(),
            system_machine=platform.machine(),
            cpu_count=platform.os.cpu_count() or 1,
            # Git
            git_commit=git_commit,
            git_branch=git_branch,
            git_is_dirty=git_is_dirty,
            git_remote_url=git_remote_url,
            # Random
            random_seed=config.seed,
            numpy_random_state=numpy_state_serialized,
            # Config
            full_config=config.to_dict(),
        )
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Obtiene el commit hash actual."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def _get_git_branch() -> Optional[str]:
        """Obtiene el branch actual."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def _is_git_dirty() -> bool:
        """Verifica si hay cambios sin commit."""
        try:
            result = subprocess.run(
                ["git", "diff", "--quiet"],
                timeout=2,
            )
            # returncode 0 = no changes, 1 = changes
            return result.returncode != 0
        except Exception:
            return False
    
    @staticmethod
    def _get_git_remote_url() -> Optional[str]:
        """Obtiene la URL del remote."""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def _serialize_numpy_state(state) -> str:
        """Serializa el estado de NumPy random."""
        # Convertir a formato serializable
        state_dict = {
            "state_type": state[0],
            "state_keys": state[1].tolist() if hasattr(state[1], 'tolist') else list(state[1]),
            "state_pos": int(state[2]),
            "state_has_gauss": int(state[3]),
            "state_cached_gauss": float(state[4]),
        }
        return json.dumps(state_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return asdict(self)
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Serializa a JSON.
        
        Args:
            path: Si se provee, guarda en archivo
        
        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkProvenance':
        """Crea instancia desde diccionario."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'BenchmarkProvenance':
        """Carga desde archivo JSON."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
    
    def is_reproducible(self) -> bool:
        """
        Verifica si el experimento es reproducible.
        
        Returns:
            True si tiene git commit y no está dirty
        """
        return (
            self.git_commit is not None
            and not self.git_is_dirty
        )
    
    def summary(self) -> str:
        """Retorna resumen legible."""
        reproducible = "✅" if self.is_reproducible() else "⚠️"
        
        summary = [
            f"Run ID: {self.run_id}",
            f"Config: {self.config_name} (hash: {self.config_hash})",
            f"Timestamp: {self.timestamp}",
            f"Python: {self.python_version.split()[0]}",
            f"NumPy: {self.numpy_version}",
            f"OS: {self.system_os} {self.system_platform}",
        ]
        
        if self.git_commit:
            summary.append(f"Git: {self.git_branch}@{self.git_commit[:8]}")
        
        summary.append(f"Reproducible: {reproducible}")
        
        return "\n".join(summary)
    
    def __str__(self) -> str:
        """String representation."""
        return self.summary()


def verify_reproducibility(provenance: BenchmarkProvenance) -> Dict[str, Any]:
    """
    Verifica si un experimento puede ser reproducido en el entorno actual.
    
    Args:
        provenance: Provenance del experimento original
    
    Returns:
        Dict con warnings y diferencias
    """
    warnings = []
    
    # Verificar Python version
    current_python = sys.version
    if current_python != provenance.python_version:
        warnings.append(
            f"Python version differs: current={current_python.split()[0]} "
            f"vs original={provenance.python_version.split()[0]}"
        )
    
    # Verificar NumPy version
    current_numpy = np.__version__
    if current_numpy != provenance.numpy_version:
        warnings.append(
            f"NumPy version differs: current={current_numpy} "
            f"vs original={provenance.numpy_version}"
        )
    
    # Verificar Git state
    current_commit = BenchmarkProvenance._get_git_commit()
    if current_commit != provenance.git_commit:
        warnings.append(
            f"Git commit differs: current={current_commit[:8] if current_commit else 'N/A'} "
            f"vs original={provenance.git_commit[:8] if provenance.git_commit else 'N/A'}"
        )
    
    # Verificar OS
    current_os = platform.system()
    if current_os != provenance.system_os:
        warnings.append(
            f"OS differs: current={current_os} vs original={provenance.system_os}"
        )
    
    return {
        "can_reproduce": len(warnings) == 0,
        "warnings": warnings,
        "critical": any("Python" in w or "NumPy" in w for w in warnings),
    }
