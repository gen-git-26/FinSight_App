# infrastructure/memory_types.py
"""
Memory system types and data classes.

Separates concerns:
- STM: Session/conversation memory
- Run Cache: Tool results within a run (A2A deduplication)
- LTM: User profiles and summaries
- RAG: Semantic search
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


class MemoryLayer(Enum):
    """Which memory layer to query."""
    STM = "stm"           # Redis session memory
    RUN_CACHE = "cache"   # Redis tool results cache
    LTM = "ltm"           # Postgres long-term
    RAG = "rag"           # Qdrant semantic


class QueryIntent(Enum):
    """Classified query intent - determines memory routing."""
    PRICE_ONLY = "price_only"           # Simple price check
    TICKER_INFO = "ticker_info"         # Stock/crypto info
    NEWS_SUMMARY = "news_summary"       # News aggregation
    TRADE_DECISION = "trade_decision"   # Full analysis
    USER_HISTORY = "user_history"       # "What did I say/do?"
    USER_PREFERENCES = "user_prefs"     # "What do I prefer?"
    SEMANTIC_SEARCH = "semantic"        # "Find similar to..."
    CONVERSATION = "conversation"       # Follow-up, context needed
    UNKNOWN = "unknown"                 # Needs fallback


@dataclass
class ClassificationResult:
    """Result of query classification."""
    intent: QueryIntent
    confidence: float  # 0.0 - 1.0
    tickers: List[str] = field(default_factory=list)
    time_range: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    layers_needed: List[MemoryLayer] = field(default_factory=list)


@dataclass
class TokenBudget:
    """Dynamic token budget based on intent."""
    conversation: int = 1500
    user_context: int = 500
    rag_results: int = 2000
    tool_results: int = 1000
    total: int = 4000

    @classmethod
    def for_intent(cls, intent: QueryIntent) -> "TokenBudget":
        """Get optimal budget for query intent."""
        budgets = {
            QueryIntent.PRICE_ONLY: cls(
                conversation=200, user_context=0, rag_results=0,
                tool_results=400, total=600
            ),
            QueryIntent.TICKER_INFO: cls(
                conversation=300, user_context=200, rag_results=500,
                tool_results=1000, total=2000
            ),
            QueryIntent.NEWS_SUMMARY: cls(
                conversation=300, user_context=200, rag_results=1500,
                tool_results=500, total=2500
            ),
            QueryIntent.TRADE_DECISION: cls(
                conversation=1000, user_context=800, rag_results=2000,
                tool_results=1200, total=5000
            ),
            QueryIntent.USER_HISTORY: cls(
                conversation=2000, user_context=500, rag_results=500,
                tool_results=0, total=3000
            ),
            QueryIntent.USER_PREFERENCES: cls(
                conversation=500, user_context=1500, rag_results=0,
                tool_results=0, total=2000
            ),
            QueryIntent.SEMANTIC_SEARCH: cls(
                conversation=500, user_context=300, rag_results=2500,
                tool_results=200, total=3500
            ),
            QueryIntent.CONVERSATION: cls(
                conversation=1500, user_context=500, rag_results=1000,
                tool_results=500, total=3500
            ),
        }
        return budgets.get(intent, cls())


@dataclass
class MemoryContext:
    """Aggregated memory context for a query."""
    # Conversation context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # User context
    user_profile: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    ticker_history: List[Dict[str, Any]] = field(default_factory=list)

    # RAG results
    rag_chunks: List[Dict[str, Any]] = field(default_factory=list)

    # Cached tool results
    cached_results: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    classification: Optional[ClassificationResult] = None
    budget_used: int = 0
    latency_ms: float = 0.0
    layers_hit: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format as prompt context string."""
        parts = []

        if self.user_preferences:
            parts.append(f"User Preferences: {self.user_preferences}")

        if self.ticker_history:
            parts.append(f"Recent decisions: {len(self.ticker_history)} records")

        if self.conversation_history:
            history_str = "\n".join([
                f"{m.get('role', 'unknown')}: {m.get('content', '')[:200]}"
                for m in self.conversation_history[-5:]
            ])
            parts.append(f"Recent conversation:\n{history_str}")

        if self.rag_chunks:
            chunks_str = "\n---\n".join([
                c.get("text", "")[:500] for c in self.rag_chunks[:3]
            ])
            parts.append(f"Relevant context:\n{chunks_str}")

        return "\n\n".join(parts)


@dataclass
class RunCacheKey:
    """Key for tool result caching within a run."""
    run_id: str
    tool_name: str
    ticker: Optional[str] = None
    params_hash: Optional[str] = None

    def to_redis_key(self) -> str:
        """Generate Redis key."""
        parts = ["run", self.run_id, self.tool_name]
        if self.ticker:
            parts.append(self.ticker)
        if self.params_hash:
            parts.append(self.params_hash)
        return ":".join(parts)


@dataclass
class UserSummary:
    """Pre-computed user summary for fast retrieval."""
    user_id: str
    risk_tolerance: Optional[str] = None  # conservative, moderate, aggressive
    preferred_sectors: List[str] = field(default_factory=list)
    avg_position_size: Optional[float] = None
    trading_style: Optional[str] = None  # day, swing, long-term
    total_decisions: int = 0
    last_active: Optional[datetime] = None
    version: int = 0  # For cache invalidation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "risk_tolerance": self.risk_tolerance,
            "preferred_sectors": self.preferred_sectors,
            "avg_position_size": self.avg_position_size,
            "trading_style": self.trading_style,
            "total_decisions": self.total_decisions,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "version": self.version,
        }


@dataclass
class TickerSummary:
    """Pre-computed ticker summary per user."""
    user_id: str
    ticker: str
    total_analyses: int = 0
    last_decision: Optional[str] = None  # buy, sell, hold
    last_analysis_date: Optional[datetime] = None
    avg_sentiment: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ticker": self.ticker,
            "total_analyses": self.total_analyses,
            "last_decision": self.last_decision,
            "last_analysis_date": self.last_analysis_date.isoformat() if self.last_analysis_date else None,
            "avg_sentiment": self.avg_sentiment,
            "notes": self.notes,
        }
