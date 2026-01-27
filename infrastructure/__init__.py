# infrastructure/__init__.py
"""
Infrastructure Layer for FinSight.

Components:
- Redis: Short-Term Memory (STM) - session cache, fast access
- PostgreSQL: Long-Term Memory (LTM) - persistent storage
- Qdrant: Vector Database - embeddings, semantic search
- Loguru: Structured logging
"""
from .redis_stm import RedisSTM, get_stm
from .postgres_ltm import PostgresLTM, get_ltm
from .logging import setup_logging, get_logger

__all__ = [
    "RedisSTM",
    "get_stm",
    "PostgresLTM",
    "get_ltm",
    "setup_logging",
    "get_logger",
]
