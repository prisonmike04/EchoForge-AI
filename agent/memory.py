from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionMemory:
    def __init__(self, memory_file: Path):
        self.memory_file = memory_file
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.memory_file.exists():
            self._history = []
            return

        try:
            self._history = json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            self._history = []

    def add(self, kind: str, payload: dict[str, Any]) -> None:
        self._history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "kind": kind,
                "payload": payload,
            }
        )
        self._persist()

    def list(self) -> list[dict[str, Any]]:
        return self._history

    def last_n(self, n: int = 8) -> list[dict[str, Any]]:
        return self._history[-n:]

    def _persist(self) -> None:
        self.memory_file.write_text(
            json.dumps(self._history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
