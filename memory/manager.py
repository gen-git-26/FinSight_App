from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from rag.embeddings import embed_texts
from rag.qdrant_client import HybridQdrant

logger = logging.getLogger(__name__)


class SessionMemory:
    def __init__(self) -> None:
        self.turns: List[Dict[str, Any]] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content, "ts": time.time()})
        if len(self.turns) > 25:
            self.turns.pop(0)

    def context(self) -> List[Dict[str, Any]]:
        return self.turns[-12:]


async def persist_summary(user_id: str, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    text = json.dumps(history)[-8000:]
    emb = (await embed_texts([text]))[0]
    payload = {"user": user_id, "type": "memory", "text": text}
    qdr = HybridQdrant()
    qdr.ensure_collections()
    qdr.upsert(points=[(f"mem-{int(time.time())}", emb, payload, text)], is_memory=True)