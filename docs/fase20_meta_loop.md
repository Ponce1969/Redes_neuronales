# Fase 20 — Meta-Learning Loop (Plan de Implementación)

## 🎯 Objetivo
Implementar un controlador de meta-aprendizaje ligero que observe pérdidas, activaciones y atención para ajustar dinámicamente:

- **Learning rate** del optimizador principal
- **Frecuencia de consolidación** (sleep cycles) del `MemoryReplaySystem`
- **Peso de atención global** o foco cognitivo

La meta es lograr que el sistema se autorregule sin entrenar un modelo aparte.

## 🧱 Componentes Propuestos

```
src/core/
 ├─ meta/
 │   ├─ rules.py              # Estrategias adaptativas (LR, foco, sueño)
 │   └─ meta_controller.py    # Controlador que aplica las reglas
 ├─ training/
 │   └─ trainer.py            # Ajustes menores para exponer lr dinámico
examples/
 └─ meta_learning_demo.py     # Demo integrando el meta-loop
```

### 1. `src/core/meta/rules.py`
Funciones puras que reciben métricas y devuelven hiperparámetros ajustados. Primera iteración:

- `adaptive_lr(prev_loss, curr_loss, lr)`
- `adaptive_focus(att_mean, focus)`
- `adaptive_sleep(loss_trend, base_interval)`

Agregar clips para mantener los valores dentro de rangos razonables.

### 2. `src/core/meta/meta_controller.py`
Clase `MetaLearningController` que:

1. Guarda referencias al `GraphTrainer`, `MemoryReplaySystem` y `CognitiveMonitor`.
2. Registra estado actual (`lr`, `focus`, `sleep_interval`, `prev_loss`).
3. Expone `observe_and_adjust(epoch, curr_loss)` para aplicar las reglas.
4. Expone `maybe_sleep(epoch)` para ejecutar `sleep_and_replay()` cuando corresponde.
5. Emite logs con nivel `META` usando el monitor para fácil depuración.

### 3. Demo `examples/meta_learning_demo.py`
Probar con un `CognitiveGraphHybrid` sencillo (input → reasoner → decision). Bucle de ~60 épocas:

1. Ejecutar `train_step` con dataset XOR.
2. Registrar pérdida en el monitor.
3. Invocar el meta-controller.
4. Mostrar métricas finales (`lr`, `focus`, `sleep_interval`).

## ✅ Entregables Esperados

- Nuevos módulos `src/core/meta/*` con tests básicos (al menos validación de clips y ajustes).
- Demo funcional que imprima los ajustes META y demuestre consolidación adaptativa.
- Actualización de `docs/proyecto.md` para reflejar la Fase 20 una vez completada.
- (Opcional) Persistir `lr`, `focus`, `sleep_interval` en el snapshot para el dashboard.

## 🗓️ Plan de Trabajo (Siguiente Sesión)

1. Crear paquete `core.meta` y definir reglas adaptativas con cobertura de tests mínimos.
2. Implementar `MetaLearningController` y asegurar compatibilidad con `GraphTrainer`/`MemoryReplaySystem`.
3. Desarrollar `meta_learning_demo.py` con dataset pequeño (XOR) y logs demostrativos.
4. (Si da el tiempo) Integrar métricas META en `dashboard_state.json` para futura visualización.

Listo para arrancar mañana con esta guía.
