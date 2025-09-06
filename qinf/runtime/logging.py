from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json, time

from qinf.memory.store import MemoryStore


@dataclass
class Logger:
    """Simple JSONL event logger with optional ledger snapshots."""
    root: Path
    run_id: str
    ledger_mode: bool = False
    memory: MemoryStore | None = None

    def __post_init__(self) -> None:
@dataclass
class Logger:
    root: Path
    run_id: str

    def __post_init__(self):
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "events.jsonl").touch(exist_ok=True)

    def log(self, **kv: Any) -> None:
        kv = {"t": time.time(), **kv}
        with (self.dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")
        if self.ledger_mode and self.memory is not None:
            self.memory.commit_snapshot(kv)
