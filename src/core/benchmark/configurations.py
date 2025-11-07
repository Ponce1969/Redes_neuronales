"""
Configuraciones de Benchmark - Sistema profesional con reproducibilidad total.

Define configuraciones completas y rastreables para experimentos científicos.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List, Dict, Any
import hashlib
import json
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """
    Configuración completa y reproducible para un benchmark.
    
    Características:
    - ✅ Hashing único para identificación
    - ✅ Seed control para reproducibilidad
    - ✅ Versionado
    - ✅ Metadata completa
    - ✅ Serialización JSON/YAML
    """
    
    # ========================================================================
    # Identidad
    # ========================================================================
    name: str
    description: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    # ========================================================================
    # Reproducibilidad
    # ========================================================================
    seed: int = 42
    deterministic: bool = True  # Forzar determinismo en NumPy
    
    # ========================================================================
    # Reasoner Configuration
    # ========================================================================
    reasoner_mode: Literal["softmax", "topk", "threshold"] = "softmax"
    n_inputs: int = 24
    n_hidden: int = 48
    n_blocks: int = 3
    topk_value: int = 2  # Para mode="topk"
    threshold_value: float = 0.5  # Para mode="threshold"
    
    # ========================================================================
    # Curriculum Configuration
    # ========================================================================
    use_curriculum: bool = True
    curriculum_type: Literal["standard", "fast", "advanced", "custom"] = "standard"
    max_epochs_per_stage: int = 50
    success_threshold: float = 0.05
    fail_threshold: float = 0.3
    
    # ========================================================================
    # Evolution Strategy
    # ========================================================================
    evolution_strategy: Literal["mutation", "crossover", "hybrid"] = "mutation"
    mutation_scale: float = 0.03
    population_size: int = 1  # Para future: evolution con población
    elite_ratio: float = 0.2  # % de mejores a mantener
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    batch_size: int = 16
    n_runs: int = 5  # Múltiples runs para validez estadística
    max_total_epochs: int = 500  # Límite global
    early_stopping_patience: int = 20
    
    # ========================================================================
    # Task Configuration
    # ========================================================================
    task_subset: Optional[List[str]] = None  # None = todas las tareas
    task_samples: int = 32  # Samples por tarea
    train_test_split: float = 0.8
    
    # ========================================================================
    # Performance
    # ========================================================================
    parallel: bool = False  # Future: paralelización
    n_workers: int = 1
    use_gpu: bool = False  # Future: aceleración GPU
    
    # ========================================================================
    # Logging & Saving
    # ========================================================================
    log_interval: int = 10
    save_checkpoints: bool = True
    save_gates_history: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if self.n_runs < 1:
            raise ValueError("n_runs debe ser >= 1")
        
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split debe estar en (0, 1)")
        
        if self.reasoner_mode == "topk" and self.topk_value < 1:
            raise ValueError("topk_value debe ser >= 1")
    
    def hash(self) -> str:
        """
        Genera hash único basado en parámetros relevantes.
        
        Returns:
            Hash SHA256 truncado (12 chars)
        """
        # Solo usar parámetros que afectan el resultado
        relevant_params = {
            "reasoner_mode": self.reasoner_mode,
            "n_hidden": self.n_hidden,
            "n_blocks": self.n_blocks,
            "use_curriculum": self.use_curriculum,
            "curriculum_type": self.curriculum_type,
            "evolution_strategy": self.evolution_strategy,
            "mutation_scale": self.mutation_scale,
            "seed": self.seed,
            "max_epochs_per_stage": self.max_epochs_per_stage,
        }
        
        config_str = json.dumps(relevant_params, sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario.
        
        Returns:
            Diccionario con todos los campos + hash
        """
        data = asdict(self)
        data["config_hash"] = self.hash()
        return data
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """
        Crea instancia desde diccionario.
        
        Args:
            data: Diccionario con configuración
        
        Returns:
            BenchmarkConfig
        """
        # Remover hash si existe (se recalcula)
        data_copy = data.copy()
        data_copy.pop("config_hash", None)
        
        return cls(**data_copy)
    
    @classmethod
    def from_json(cls, path: Path) -> 'BenchmarkConfig':
        """
        Carga desde archivo JSON.
        
        Args:
            path: Ruta al archivo JSON
        
        Returns:
            BenchmarkConfig
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation legible."""
        return (
            f"BenchmarkConfig(\n"
            f"  name={self.name!r},\n"
            f"  hash={self.hash()},\n"
            f"  curriculum={self.use_curriculum},\n"
            f"  mode={self.reasoner_mode},\n"
            f"  n_runs={self.n_runs}\n"
            f")"
        )


# ============================================================================
# Configuraciones Pre-Definidas
# ============================================================================

def create_baseline_random() -> BenchmarkConfig:
    """Baseline: Sin curriculum, modo aleatorio."""
    return BenchmarkConfig(
        name="baseline_random",
        description="Random baseline sin curriculum learning",
        use_curriculum=False,
        max_total_epochs=200,
        seed=42,
        tags=["baseline", "random"],
    )


def create_curriculum_softmax() -> BenchmarkConfig:
    """Curriculum learning con gates softmax."""
    return BenchmarkConfig(
        name="curriculum_softmax",
        description="Curriculum learning estándar con softmax gates",
        use_curriculum=True,
        curriculum_type="standard",
        reasoner_mode="softmax",
        seed=42,
        tags=["curriculum", "softmax"],
    )


def create_curriculum_topk() -> BenchmarkConfig:
    """Curriculum learning con gates top-k."""
    return BenchmarkConfig(
        name="curriculum_topk",
        description="Curriculum learning con top-k gate selection",
        use_curriculum=True,
        curriculum_type="standard",
        reasoner_mode="topk",
        topk_value=2,
        seed=42,
        tags=["curriculum", "topk"],
    )


def create_no_curriculum_topk() -> BenchmarkConfig:
    """Sin curriculum, solo top-k gates."""
    return BenchmarkConfig(
        name="no_curriculum_topk",
        description="Top-K gates sin curriculum progresivo",
        use_curriculum=False,
        reasoner_mode="topk",
        topk_value=2,
        max_total_epochs=200,
        seed=42,
        tags=["no_curriculum", "topk"],
    )


def create_curriculum_fast() -> BenchmarkConfig:
    """Curriculum rápido para pruebas."""
    return BenchmarkConfig(
        name="curriculum_fast",
        description="Curriculum rápido (4 etapas, pocas epochs)",
        use_curriculum=True,
        curriculum_type="fast",
        max_epochs_per_stage=30,
        n_runs=3,
        seed=42,
        tags=["curriculum", "fast", "test"],
    )


def create_high_mutation() -> BenchmarkConfig:
    """Curriculum con mutación agresiva."""
    return BenchmarkConfig(
        name="high_mutation",
        description="Curriculum con alta tasa de mutación",
        use_curriculum=True,
        curriculum_type="standard",
        mutation_scale=0.1,  # 3x normal
        seed=42,
        tags=["curriculum", "high_mutation"],
    )


def create_large_reasoner() -> BenchmarkConfig:
    """Reasoner grande para tareas complejas."""
    return BenchmarkConfig(
        name="large_reasoner",
        description="Reasoner con más capacidad (hidden=96)",
        use_curriculum=True,
        curriculum_type="advanced",
        n_hidden=96,
        seed=42,
        tags=["curriculum", "large"],
    )


# ============================================================================
# Registry de Configuraciones
# ============================================================================

BENCHMARK_CONFIGS: Dict[str, BenchmarkConfig] = {
    "baseline_random": create_baseline_random(),
    "curriculum_softmax": create_curriculum_softmax(),
    "curriculum_topk": create_curriculum_topk(),
    "no_curriculum_topk": create_no_curriculum_topk(),
    "curriculum_fast": create_curriculum_fast(),
    "high_mutation": create_high_mutation(),
    "large_reasoner": create_large_reasoner(),
}


def get_config(name: str) -> BenchmarkConfig:
    """
    Obtiene configuración por nombre.
    
    Args:
        name: Nombre de la configuración
    
    Returns:
        BenchmarkConfig
    
    Raises:
        KeyError: Si no existe la configuración
    """
    if name not in BENCHMARK_CONFIGS:
        available = list(BENCHMARK_CONFIGS.keys())
        raise KeyError(
            f"Config '{name}' no encontrada. Disponibles: {available}"
        )
    
    return BENCHMARK_CONFIGS[name]


def list_configs() -> List[str]:
    """Lista todas las configuraciones disponibles."""
    return list(BENCHMARK_CONFIGS.keys())


def create_custom_config(**kwargs) -> BenchmarkConfig:
    """
    Crea configuración personalizada.
    
    Args:
        **kwargs: Parámetros para BenchmarkConfig
    
    Returns:
        BenchmarkConfig personalizada
    """
    # Defaults razonables
    defaults = {
        "name": "custom",
        "description": "Configuración personalizada",
        "seed": 42,
    }
    
    defaults.update(kwargs)
    return BenchmarkConfig(**defaults)
