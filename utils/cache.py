from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Optional


class FileTTLCache:
    def __init__(self, cache_dir: str, ttl_seconds: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _key_to_path(self, key: str) -> Path:
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if time.time() - obj.get("ts", 0) > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return obj.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        tmp = {
            "ts": time.time(),
            "value": value,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(tmp, f)
