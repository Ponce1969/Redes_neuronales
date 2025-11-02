from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional


class CognitiveLogger:
    """Logger cognitivo con soporte para archivo JSON y buffer en memoria."""

    def __init__(self, to_file: bool = True, file_path: str = "logs/cognitive_log.json") -> None:
        self.to_file = to_file
        self.file_path = file_path
        self.events: List[Dict[str, object]] = []
        if self.to_file:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def log(self, level: str, message: str, data: Optional[Dict[str, object]] = None) -> None:
        event = {
            "time": time.strftime("%H:%M:%S"),
            "level": level.upper(),
            "message": message,
            "data": data or {},
        }
        self.events.append(event)
        print(f"[{event['time']}][{event['level']}] {event['message']}")
        if self.to_file:
            self._flush_to_file(event)

    def _flush_to_file(self, event: Dict[str, object]) -> None:
        with open(self.file_path, "a", encoding="utf-8") as file:
            json.dump(event, file, ensure_ascii=False)
            file.write("\n")
