from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import uuid
import lmdb

from .store import MemoryStore


@dataclass
class LMDBStore(MemoryStore):
    """MemoryStore backed by an LMDB key-value database."""

    path: Path
    map_size: int = 1 << 30  # 1GB default

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(str(self.path), map_size=self.map_size, subdir=True, max_dbs=1)

    def put(self, key: bytes, value: bytes) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def get(self, key: bytes) -> bytes | None:
        with self.env.begin(write=False) as txn:
            return txn.get(key)

    def commit_snapshot(self, manifest: dict) -> str:
        snap_id = uuid.uuid4().hex
        data = json.dumps(manifest).encode("utf-8")
        self.put(f"snapshot:{snap_id}".encode("utf-8"), data)
        return snap_id
