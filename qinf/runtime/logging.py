from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json, time

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
