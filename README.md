# 🧠 Neural Core

Motor neuronal modular en Python puro.

Este proyecto busca construir una **inteligencia artificial desde cero**,  
empezando por **microneuronas**, pasando por **macroneuronas**,  
hasta redes capaces de **aprender, predecir y evolucionar**.

---

## 🚀 Estructura

src/
├── core/ # Lógica neuronal pura
├── engine/ # Entrenamiento e inferencia
├── io/ # Entrada/salida, logs, persistencia
└── app.py # Punto de entrada principal



---

## 🧩 Objetivos

1. Construir un modelo neuronal propio (sin frameworks externos).
2. Implementar funciones de activación, pérdidas y optimización.
3. Permitir aprendizaje y predicción desde cero.
4. Evolucionar hacia módulos inteligentes (memoria, razonamiento, percepción).



✅ 5️⃣ Prueba rápida de entorno

Una vez tengas todo creado, ejecuta:

uv sync
uv run pytest


🚀 6️⃣ Próximo paso

Ahora que tenemos el esqueleto, el siguiente paso será la Fase 1: Microneuronas y Activaciones.
Allí crearemos:

core/activations.py: funciones (sigmoid, relu, tanh, etc.)

core/neuron.py: clase Neuron con pesos, bias y función de activación

core/utils.py: generador de pesos aleatorios y normalización