from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    with _LOCK:
        with open(path, "w", encoding="utf-8") as f:
            f.write(serialized)
