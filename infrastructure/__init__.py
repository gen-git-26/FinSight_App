# infrastructure/__init__.py
"""
Infrastructure Layer for FinSight.

Components:
- MemoryManager: Unified interface with smart routing
- Redis STM: Session/conversation memory
- RunCache: Tool results caching (A2A deduplication)
- PostgreSQL LTM: Long-term storage
- PostgresSummaries: Pre-computed user summaries
- QueryClassifier: 2-stage query classification
- Qdrant: Vector Database - embeddings, semantic search
- Loguru: Structured logging

Usage:
    from infrastructure import get_memory_manager

    manager = get_memory_manager()
    context = await manager.get_context(
        query="What's the price of AAPL?",
        session_id="sess_123",
        user_id="user_456",
        run_id="run_789"
    )
"""
from .memory_types import (
    MemoryLayer,
    QueryIntent,
    ClassificationResult,
    TokenBudget,
    MemoryContext,
    RunCacheKey,
    UserSummary,
    TickerSummary,
)
from .memory_manager import MemoryManager, get_memory_manager
from .redis_stm import RedisSTM, get_stm
from .run_cache import RunCache, get_run_cache
from .postgres_summaries import PostgresSummaries, get_summaries
from .postgres_ltm import PostgresLTM, get_ltm
from .query_classifier import QueryClassifier, get_classifier
from .logging import setup_logging, get_logger

__all__ = [
    # Types
    "MemoryLayer",
    "QueryIntent",
    "ClassificationResult",
    "TokenBudget",
    "MemoryContext",
    "RunCacheKey",
    "UserSummary",
    "TickerSummary",
    # Main interface
    "MemoryManager",
    "get_memory_manager",
    # Components
    "RedisSTM",
    "get_stm",
    "RunCache",
    "get_run_cache",
    "PostgresSummaries",
    "get_summaries",
    "PostgresLTM",
    "get_ltm",
    "QueryClassifier",
    "get_classifier",
    # Logging
    "setup_logging",
    "get_logger",
]
