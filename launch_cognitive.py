"""Utilidad para lanzar entrenamiento y dashboard en paralelo.

Ejecutar con:
    PYTHONPATH=src uv run python launch_cognitive.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
DASHBOARD_PATH = ROOT / "dashboard" / "app_dashboard.py"
DEMO_PATH = ROOT / "examples" / "memory_replay_demo.py"


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(SRC_PATH))
    env.setdefault("STREAMLIT_SERVER_PORT", "8501")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return env


def _launch_training(env: dict[str, str]) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, str(DEMO_PATH)]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def _launch_dashboard(env: dict[str, str]) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH), "--server.headless=true"]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    env = _build_env()

    print("[launcher] Iniciando demo de entrenamiento…")
    training_proc = _launch_training(env)

    # breve espera para que el monitor empiece a generar datos
    time.sleep(3)

    print("[launcher] Iniciando dashboard Streamlit en http://localhost:8501 …")
    dashboard_proc = _launch_dashboard(env)

    try:
        training_exit = training_proc.wait()
        print(f"[launcher] Proceso de entrenamiento finalizó con código {training_exit}.")
    except KeyboardInterrupt:
        print("[launcher] Interrupción recibida. Terminando procesos…")
    finally:
        for proc in (training_proc, dashboard_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
