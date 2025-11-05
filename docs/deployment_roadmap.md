# Cognitive Deployment Roadmap

Guía estratégica para desplegar la plataforma cognitiva en entorno local, Orange Pi 5 Plus y un futuro nodo cloud. Resume arquitectura, configuración y pasos concretos para llegar a producción con un mismo repositorio.

---

## 1. Capas del sistema

| Capa | Rol | Componentes clave |
|------|-----|-------------------|
| 🧠 **Core Cognitivo** | Neuronas, grafos, memoria, scheduler, evolución | `src/core/*` |
| ⚙️ **Infraestructura / API** | FastAPI, persistencia, federación, endpoints REST | `src/api/*`, `src/persistence/*` |
| 📊 **Interfaz / Monitoreo** | Dashboard Streamlit, métricas cognitivas | `dashboard/*` |

Cada despliegue utiliza las tres capas, pero habilita componentes distintos según el rol del nodo.

---

## 2. Roles y objetivos por entorno

| Entorno | Rol | Qué ejecuta | Objetivo |
|---------|-----|-------------|----------|
| 💻 **PC Local** | Nodo de desarrollo y control | Core + API + Dashboard | Experimentar, visualizar, depurar |
| 🤖 **Orange Pi 5 Plus** | Nodo cognitivo entrenador | Core + API + Scheduler | Entrenar, evolucionar, exponer métricas |
| ☁️ **Servidor Cloud** *(futuro)* | Nodo federador y coordinador | API + Federation Server + Storage | Promediar modelos de múltiples agentes |

Cada nodo utiliza la misma base de código. Las diferencias se controlan vía variables de entorno y procesos lanzados.

---

## 3. Estructura de proyecto (idéntica en todos los nodos)

```text
/home/<usuario>/neural_core/
├── src/
├── tests/
├── dashboard/
├── docs/
├── pyproject.toml
└── uv.lock
```

Clona el repositorio completo en cada entorno. Ajusta únicamente la configuración (``.env`` y servicios en ejecución).

---

## 4. Variables de entorno sugeridas

| Variable | PC Local | Orange Pi 5 | Servidor Cloud |
|----------|----------|-------------|----------------|
| `ROLE` | `controller` | `agent` | `federator` |
| `SCHEDULER_ENABLED` | `false` | `true` | `false` |
| `API_URL` | `http://localhost:8000` | `https://<tunnel-pi>` | `https://<cloud-host>` |
| `FEDERATION_URL` | `https://<tunnel-pi>` | `https://<cloud-host>` | `https://<cloud-host>` |
| `API_KEY` | misma clave compartida entre nodos |

> Cambia `FEDERATION_URL` al endpoint del nodo federador cuando el servidor cloud esté activo.

---

## 5. Flujo de comunicación

1. **Backend unificado**: todos los nodos ejecutan `uvicorn src.api.server:app`.
2. **Exposición segura**: la Orange Pi publica su API mediante Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`).
3. **Panel central**: el dashboard Streamlit (en la PC local) consulta métricas y estado del Pi a través del túnel.
4. **Scheduler activo**: únicamente en la Pi (u otros agentes). Ejecuta entrenamiento continuo, persistencia, federación, evolución y sueño cognitivo.
5. **Federación** *(cuando el cloud esté listo)*: cada agente envía pesos al servidor federador (`/federate/upload`) y recibe el promedio global (`/federate/global`).

---

## 6. Pasos concretos por entorno

### 💻 PC Local (controller)
1. Clonar repositorio y crear `.env` con valores de la tabla.
2. Instalar dependencias (`uv sync`).
3. Levantar el backend: `uv run uvicorn src.api.server:app --reload`.
4. Iniciar dashboard: `uv run streamlit run dashboard/dashboard_latent.py`.
5. Configurar `FEDERATION_URL` con la URL pública de la Pi para consumir métricas remotas.
6. Verificar interacción desde Streamlit.

### 🤖 Orange Pi 5 Plus (agent)
1. Clonar el repositorio y copiar `.env` con `ROLE=agent` y `SCHEDULER_ENABLED=true`.
2. Instalar dependencias (`uv sync` o `pip install -r requirements.txt`).
3. Lanzar backend: `uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000`.
4. Mantener Cloudflare Tunnel activo: `cloudflared tunnel --url http://localhost:8000`.
5. Confirmar que el scheduler corre (logs `[Scheduler]` en consola) y expone métricas `/status`, `/federate/*` si corresponde.

### ☁️ Servidor Cloud (federator, futuro)
1. Clonar repositorio, `.env` con `ROLE=federator` y `SCHEDULER_ENABLED=false`.
2. Desplegar backend (FastAPI) detrás de HTTPS (NGINX/Reverse Proxy o Managed FastAPI hosting).
3. Habilitar almacenamiento persistente (S3, GCS o volumen local) para pesos agregados.
4. Verificar endpoints `/federate/upload` y `/federate/global`.
5. Configurar agentes (Pi, PC) con `FEDERATION_URL=https://<cloud-host>`.

---

## 7. Checklist antes de producción

- [ ] Claves `API_KEY` consistentes entre nodos.
- [ ] `.env` definido según rol.
- [ ] Persistencia (`core.persistence`) confirmada en cada nodo.
- [ ] Cloudflare Tunnel operativo en la Pi (url registrada en el `.env` de la PC).
- [ ] Tests locales (`uv run pytest`) verdes.
- [ ] Scheduler activo solo donde corresponde (`SCHEDULER_ENABLED=true`).
- [ ] Monitor Streamlit accediendo sin errores.
- [ ] (Cuando aplique) Servidor federador respondiendo en HTTPS.

---

## 8. Próximos pasos hacia fase 26

1. Automatizar despliegues (systemd service / Docker / supervisord según entorno).
2. Añadir observabilidad (logs centralizados, alertas de pérdida elevada, disponibilidad del túnel).
3. Integrar servidor cloud y validar flujo federado end-to-end.
4. Documentar incident response y backups (weights/memories/uv.lock).
5. Formalizar pipelines CI/CD para tests y despliegues en los nodos.

---

Con esta hoja de ruta, cada entorno sabe qué ejecutar, cómo configurarse y cómo interactuar con el resto. Sirve como guía operativa para mantener viva la red cognitiva y escalarla a producción de forma ordenada.
