# memory/manager.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from rag.embeddings import embed_texts
from rag.qdrant_client import HybridQdrant
from qdrant_client.http import models as rest
from utils.config import load_settings

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

_session = SessionMemory()


def _user_id() -> str:
    cfg = load_settings()
    return getattr(cfg, 'user_id', None) or "default"


async def persist_summary(user_id: str, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    text = json.dumps(history, ensure_ascii=False)[-8000:]
    qdr = HybridQdrant()
    qdr.ensure_collections()
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    items = [{
        "id": f"mem-{int(time.time())}",
        "text": text,
        "type": "memory",
        "user": user_id,
        "date": now,
        "source": "memory",
    }]
    await qdr.upsert_snippets(items)


async def fetch_memory(query: str, k: int = 3) -> str:
    from rag.fusion import retrieve
    qdr = HybridQdrant()
    qdr.ensure_collections()
    flt = [
        rest.FieldCondition(key='type', match=rest.MatchAny(any=['memory'])),
        rest.FieldCondition(key='user', match=rest.MatchAny(any=[_user_id()])),
    ]
    docs = await retrieve(query, filters=flt, k=k)
    parts = [str((d.payload or {}).get('text', ''))[:600] for d in docs]
    return "\n".join(parts)


async def persist_turn(user_text: str, assistant_text: str) -> None:
    _session.add("user", user_text)
    _session.add("assistant", assistant_text)
    # periodically persist a compact summary of the last turns
    if len(_session.turns) % 4 == 0:
        await persist_summary(_user_id(), _session.context())