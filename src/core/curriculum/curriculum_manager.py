"""
Curriculum Manager - Gestión profesional de aprendizaje progresivo.

Manager principal que coordina el entrenamiento curriculum del Reasoner
sin variables globales, con checkpointing automático y métricas avanzadas.
"""

import time
import threading
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path

from core.curriculum.curriculum_stage import CurriculumStage
from core.curriculum.evaluator import CurriculumEvaluator
from core.curriculum.checkpointer import CurriculumCheckpointer
from core.curriculum.metrics import CognitiveMetrics


class CurriculumManager:
    """
    Gestiona el aprendizaje progresivo del Reasoner.
    
    Características:
    - ✅ Sin variables globales (usa inyección de dependencias)
    - ✅ Checkpointing automático
    - ✅ Métricas avanzadas
    - ✅ Resume desde última etapa
    - ✅ Thread-safe
    - ✅ Early stopping inteligente
    
    Uso:
        manager = CurriculumManager(reasoner_manager, graph)
        manager.add_stage(stage1)
        manager.add_stage(stage2)
        manager.run()
    """
    
    def __init__(
        self,
        reasoner_manager,
        graph,
        checkpoint_dir: str = "data/curriculum",
        auto_save: bool = True,
    ):
        """
        Inicializa el Curriculum Manager.
        
        Args:
            reasoner_manager: Instancia de ReasonerManager (inyección de dependencia)
            graph: Instancia de CognitiveGraphHybrid
            checkpoint_dir: Directorio para guardar checkpoints
            auto_save: Si True, guarda automáticamente después de cada etapa
        """
        self.reasoner_manager = reasoner_manager
        self.graph = graph
        self.auto_save = auto_save
        
        # Componentes
        self.evaluator = CurriculumEvaluator(graph)
        self.checkpointer = CurriculumCheckpointer(checkpoint_dir)
        
        # Estado
        self.stages: List[CurriculumStage] = []
        self.current_stage_idx: int = 0
        self.history: List[Dict[str, Any]] = []
        self.running: bool = False
        self.paused: bool = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Intentar cargar estado previo
        self._load_checkpoint()
    
    def add_stage(self, stage: CurriculumStage):
        """
        Añade una etapa al curriculum.
        
        Args:
            stage: CurriculumStage a añadir
        """
        with self.lock:
            self.stages.append(stage)
            print(f"📚 Etapa añadida: {stage.name} (dificultad {stage.difficulty})")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Ejecuta el curriculum completo desde la etapa actual.
        
        Returns:
            Historial completo de performance por etapa
        """
        with self.lock:
            if self.running:
                raise RuntimeError("El curriculum ya está corriendo")
            
            self.running = True
            self.paused = False
        
        print(f"\n🚀 Iniciando Curriculum Learning")
        print(f"   Etapas: {len(self.stages)}")
        print(f"   Comenzando desde: {self.current_stage_idx + 1}")
        print(f"   Auto-save: {'✅' if self.auto_save else '❌'}\n")
        
        try:
            for idx in range(self.current_stage_idx, len(self.stages)):
                # Verificar si fue pausado
                if self.paused:
                    print("⏸️  Curriculum pausado")
                    break
                
                stage = self.stages[idx]
                success = self._run_stage(stage, idx)
                
                if not success:
                    print(f"\n❌ Curriculum detenido en etapa '{stage.name}'")
                    break
                
                # Avanzar al siguiente stage
                with self.lock:
                    self.current_stage_idx = idx + 1
                
                # Auto-save checkpoint
                if self.auto_save:
                    self._save_checkpoint()
        
        finally:
            with self.lock:
                self.running = False
        
        print("\n🏁 Curriculum completado!")
        self._print_summary()
        
        return self.history
    
    def _run_stage(self, stage: CurriculumStage, idx: int) -> bool:
        """
        Ejecuta una etapa individual del curriculum.
        
        Args:
            stage: CurriculumStage a ejecutar
            idx: Índice de la etapa
        
        Returns:
            True si completó exitosamente, False si falló
        """
        print(f"\n{'='*60}")
        print(f"🎯 ETAPA {idx + 1}/{len(self.stages)}: {stage.name.upper()}")
        print(f"   Dificultad: {stage.difficulty}/10")
        print(f"   Success threshold: {stage.success_threshold:.4f}")
        print(f"   Max epochs: {stage.max_epochs}")
        print(f"{'='*60}\n")
        
        # Generar dataset para esta etapa
        X, Y = stage.task_generator()
        print(f"📊 Dataset generado: X={X.shape}, Y={Y.shape}")
        
        best_metrics = {"mse_loss": np.inf}
        best_epoch = 0
        
        for epoch in range(stage.max_epochs):
            # Verificar pausa
            if self.paused:
                return False
            
            # Evaluar Reasoner actual
            metrics = self.evaluator.evaluate(
                self.reasoner_manager.reasoner,
                X,
                Y,
                track_gates=True,
            )
            
            # Actualizar mejor si mejoró
            if metrics['mse_loss'] < best_metrics['mse_loss']:
                best_metrics = metrics.copy()
                best_epoch = epoch
                
                # Guardar el mejor Reasoner
                if self.auto_save:
                    self.reasoner_manager.save()
            
            # Log progreso
            if epoch % stage.log_interval == 0:
                self._log_epoch(epoch, stage.max_epochs, metrics, best_metrics)
            
            # Early stopping si alcanzó el threshold
            if metrics['mse_loss'] <= stage.success_threshold:
                print(f"\n✅ ¡Etapa '{stage.name}' completada en {epoch} epochs!")
                print(f"   Best loss: {best_metrics['mse_loss']:.4f}")
                
                # Marcar etapa como completada
                stage.mark_completed(epoch, best_metrics)
                
                # Guardar en historial
                self.history.append({
                    "stage": stage.name,
                    "difficulty": stage.difficulty,
                    "epochs": epoch,
                    "best_epoch": best_epoch,
                    **best_metrics,
                })
                
                return True
            
            # Evolucionar el Reasoner (optimización ligera)
            self._evolve_reasoner(X, Y, stage)
            
            # Pequeña pausa para no saturar CPU
            time.sleep(0.01)
        
        # Terminó todas las epochs sin alcanzar success_threshold
        print(f"\n⚠️  Etapa '{stage.name}' alcanzó max_epochs ({stage.max_epochs})")
        print(f"   Best loss: {best_metrics['mse_loss']:.4f}")
        
        # Verificar si al menos pasó el fail_threshold
        if best_metrics['mse_loss'] > stage.fail_threshold:
            print(f"   ❌ FALLO: loss > fail_threshold ({stage.fail_threshold})")
            return False
        
        # Pasó parcialmente (entre success y fail threshold)
        print(f"   ⚠️  PARCIAL: Continuando al siguiente stage")
        
        stage.mark_completed(stage.max_epochs, best_metrics)
        self.history.append({
            "stage": stage.name,
            "difficulty": stage.difficulty,
            "epochs": stage.max_epochs,
            "best_epoch": best_epoch,
            "partial": True,
            **best_metrics,
        })
        
        return True
    
    def _evolve_reasoner(self, X: np.ndarray, Y: np.ndarray, stage: CurriculumStage):
        """
        Realiza evolución ligera del Reasoner.
        
        Args:
            X: Dataset de inputs
            Y: Dataset de targets
            stage: Etapa actual (contiene parámetros de evolución)
        """
        for _ in range(stage.evolution_generations):
            # Mutar Reasoner actual
            child = self.reasoner_manager.reasoner.mutate(scale=stage.mutation_scale)
            
            # Evaluar hijo
            child_loss = self.evaluator.evaluate(child, X, Y, track_gates=False)['mse_loss']
            
            # Reemplazar si es mejor
            current_loss = self.evaluator.evaluate(
                self.reasoner_manager.reasoner, X, Y, track_gates=False
            )['mse_loss']
            
            if child_loss < current_loss:
                with self.lock:
                    self.reasoner_manager.reasoner = child
    
    def _log_epoch(self, epoch: int, max_epochs: int, current: dict, best: dict):
        """Imprime progreso de la época actual."""
        metrics_str = CognitiveMetrics.format_metrics(current, precision=4)
        print(
            f"  Epoch {epoch:3d}/{max_epochs} | "
            f"{metrics_str} | "
            f"best={best['mse_loss']:.4f}"
        )
    
    def _print_summary(self):
        """Imprime resumen final del curriculum."""
        if not self.history:
            return
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL CURRICULUM")
        print("="*60)
        
        for i, record in enumerate(self.history, 1):
            status = "✅" if not record.get('partial', False) else "⚠️"
            print(
                f"{status} {i}. {record['stage']:12s} | "
                f"Dificultad: {record['difficulty']}/10 | "
                f"Epochs: {record['epochs']:3d} | "
                f"Loss: {record['mse_loss']:.4f} | "
                f"Acc: {record.get('accuracy', 0):.1%}"
            )
        
        # Métricas globales
        total_epochs = sum(r['epochs'] for r in self.history)
        avg_loss = np.mean([r['mse_loss'] for r in self.history])
        
        print("="*60)
        print(f"Total epochs: {total_epochs}")
        print(f"Avg loss: {avg_loss:.4f}")
        print(f"Etapas completadas: {len(self.history)}/{len(self.stages)}")
        print("="*60 + "\n")
    
    def pause(self):
        """Pausa la ejecución del curriculum."""
        with self.lock:
            self.paused = True
            print("⏸️  Pausando curriculum...")
    
    def resume(self):
        """Reanuda la ejecución del curriculum."""
        with self.lock:
            self.paused = False
            print("▶️  Reanudando curriculum...")
    
    def reset(self):
        """Resetea el curriculum para empezar desde cero."""
        with self.lock:
            if self.running:
                raise RuntimeError("No se puede resetear mientras está corriendo")
            
            self.current_stage_idx = 0
            self.history = []
            
            for stage in self.stages:
                stage.reset()
            
            self.checkpointer.reset()
            print("🔄 Curriculum reseteado")
    
    def status(self) -> Dict[str, Any]:
        """
        Retorna el estado actual del curriculum.
        
        Returns:
            Diccionario con métricas y estado
        """
        with self.lock:
            current_stage = None
            if 0 <= self.current_stage_idx < len(self.stages):
                current_stage = self.stages[self.current_stage_idx]
            
            return {
                "running": self.running,
                "paused": self.paused,
                "current_stage_idx": self.current_stage_idx,
                "total_stages": len(self.stages),
                "current_stage_name": current_stage.name if current_stage else None,
                "stages_completed": len(self.history),
                "history": self.history,
                "stage_names": [s.name for s in self.stages],
                "progress": (
                    self.current_stage_idx / len(self.stages) * 100
                    if self.stages else 0
                ),
            }
    
    def _save_checkpoint(self):
        """Guarda checkpoint del estado actual."""
        self.checkpointer.save(
            current_stage_idx=self.current_stage_idx,
            total_stages=len(self.stages),
            stage_names=[s.name for s in self.stages],
            history=self.history,
            reasoner_manager=self.reasoner_manager,
        )
    
    def _load_checkpoint(self):
        """Intenta cargar checkpoint previo."""
        state = self.checkpointer.load()
        
        if state:
            self.current_stage_idx = state.get('current_stage_idx', 0)
            self.history = state.get('history', [])
            print(f"♻️  Curriculum restaurado desde checkpoint")
