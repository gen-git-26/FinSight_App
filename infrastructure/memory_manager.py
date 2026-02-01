# infrastructure/memory_manager.py
"""
Memory Manager - Unified interface with intelligent routing.

Key features:
1. Query classification → route to minimal necessary layers
2. Race pattern: Redis first, cancel others if sufficient
3. Parallel fetch with timeout for multi-layer queries
4. Dynamic token budget by intent
5. Cache invalidation with versioning

Flow:
    Query → Classify → Route → Fetch (race) → Budget → Return
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field

from infrastructure.memory_types import (
    MemoryLayer,
    QueryIntent,
    ClassificationResult,
    TokenBudget,
    MemoryContext,
)
from infrastructure.query_classifier import QueryClassifier, get_classifier
from infrastructure.redis_stm import RedisSTM, get_stm
from infrastructure.run_cache import RunCache, get_run_cache
from infrastructure.postgres_summaries import PostgresSummaries, get_summaries


@dataclass
class MemoryConfig:
    """Memory manager configuration."""
    # Timeouts (ms)
    redis_timeout: int = 50
    postgres_timeout: int = 100
    qdrant_timeout: int = 150
    total_timeout: int = 200

    # Token budgets
    default_budget: int = 4000

    # Race settings
    cancel_on_sufficient: bool = True
    min_redis_sufficiency: float = 0.7  # If Redis provides 70%+ needed, skip others


class MemoryManager:
    """
    Unified memory interface with smart routing.

    Usage:
        manager = get_memory_manager()
        context = await manager.get_context(
            query="What's the price of AAPL?",
            session_id="sess_123",
            user_id="user_456",
            run_id="run_789"
        )
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        stm: Optional[RedisSTM] = None,
        cache: Optional[RunCache] = None,
        summaries: Optional[PostgresSummaries] = None,
        classifier: Optional[QueryClassifier] = None
    ):
        self.config = config or MemoryConfig()
        self.stm = stm or get_stm()
        self.cache = cache or get_run_cache()
        self.summaries = summaries or get_summaries()
        self.classifier = classifier or get_classifier()

        # Lazy load Qdrant (might not be available)
        self._qdrant = None

    @property
    def qdrant(self):
        """Lazy load Qdrant client."""
        if self._qdrant is None:
            try:
                from rag.qdrant_client import HybridQdrant
                self._qdrant = HybridQdrant()
            except Exception as e:
                print(f"[Memory] Qdrant not available: {e}")
        return self._qdrant

    # === Core: Get Context ===

    async def get_context(
        self,
        query: str,
        session_id: str,
        user_id: str,
        run_id: Optional[str] = None,
        token_budget: Optional[int] = None
    ) -> MemoryContext:
        """
        Smart retrieval - only fetches what's needed.

        1. Classifies query to determine intent
        2. Routes to minimal necessary layers
        3. Uses race pattern (Redis first, cancel if sufficient)
        4. Applies dynamic token budget
        """
        start_time = time.time()
        context = MemoryContext()

        # Step 1: Classify query
        classification = await self.classifier.classify(query)
        context.classification = classification

        # Step 2: Determine token budget
        budget = TokenBudget.for_intent(classification.intent)
        if token_budget:
            budget.total = token_budget

        # Step 3: Route and fetch
        layers_needed = set(classification.layers_needed)

        # Always check STM for conversation context
        if MemoryLayer.STM in layers_needed or classification.intent == QueryIntent.CONVERSATION:
            await self._fetch_stm(context, session_id, budget)

        # Fetch other layers based on classification
        fetch_tasks = []

        if MemoryLayer.RUN_CACHE in layers_needed and run_id:
            fetch_tasks.append(
                self._fetch_run_cache(context, run_id, classification.tickers)
            )

        if MemoryLayer.LTM in layers_needed:
            fetch_tasks.append(
                self._fetch_ltm(context, user_id, classification.tickers, budget)
            )

        if MemoryLayer.RAG in layers_needed:
            fetch_tasks.append(
                self._fetch_rag(context, query, classification, budget)
            )

        # Execute with timeout (race pattern)
        if fetch_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*fetch_tasks, return_exceptions=True),
                    timeout=self.config.total_timeout / 1000
                )
            except asyncio.TimeoutError:
                print(f"[Memory] Fetch timeout, proceeding with partial results")

        # Record metrics
        context.latency_ms = (time.time() - start_time) * 1000
        context.layers_hit = [l.value for l in layers_needed]

        return context

    # === Layer Fetchers ===

    async def _fetch_stm(
        self,
        context: MemoryContext,
        session_id: str,
        budget: TokenBudget
    ) -> None:
        """Fetch from Redis STM (session, conversation history)."""
        try:
            # Get conversation history
            history = self.stm.get_history(session_id, limit=10)
            context.conversation_history = history

            # Get context summary if exists
            summary = self.stm.get_context_summary(session_id)
            if summary:
                context.cached_results["context_summary"] = summary

        except Exception as e:
            print(f"[Memory] STM fetch error: {e}")

    async def _fetch_run_cache(
        self,
        context: MemoryContext,
        run_id: str,
        tickers: List[str]
    ) -> None:
        """Fetch cached tool results for current run."""
        try:
            cached = {}
            for ticker in tickers:
                # Check for cached data
                quote = self.cache.get_quote(run_id, ticker)
                if quote:
                    cached[f"quote:{ticker}"] = quote

                ohlcv = self.cache.get_ohlcv(run_id, ticker)
                if ohlcv:
                    cached[f"ohlcv:{ticker}"] = ohlcv

                news = self.cache.get_news(run_id, ticker)
                if news:
                    cached[f"news:{ticker}"] = news

            context.cached_results.update(cached)

        except Exception as e:
            print(f"[Memory] RunCache fetch error: {e}")

    async def _fetch_ltm(
        self,
        context: MemoryContext,
        user_id: str,
        tickers: List[str],
        budget: TokenBudget
    ) -> None:
        """Fetch from PostgreSQL LTM (user profile, history)."""
        try:
            # Check Redis cache first for user preferences
            snapshot = self.stm.get_user_snapshot(user_id)
            current_version = self.summaries.get_user_version(user_id)

            if snapshot and snapshot.get("version") == current_version:
                # Cache hit - use cached preferences
                context.user_preferences = snapshot.get("preferences")
            else:
                # Cache miss - fetch from Postgres and cache
                user_summary = self.summaries.get_user_summary(user_id)
                if user_summary:
                    context.user_profile = user_summary
                    context.user_preferences = {
                        "risk_tolerance": user_summary.get("risk_tolerance"),
                        "preferred_sectors": user_summary.get("preferred_sectors"),
                        "trading_style": user_summary.get("trading_style"),
                    }
                    # Cache in Redis with version
                    self.stm.set_user_snapshot(
                        user_id,
                        context.user_preferences,
                        current_version
                    )

            # Fetch ticker-specific history if tickers provided
            if tickers and budget.user_context > 200:
                ticker_history = []
                for ticker in tickers[:3]:  # Limit to 3 tickers
                    summary = self.summaries.get_ticker_summary(user_id, ticker)
                    if summary:
                        ticker_history.append(summary)
                context.ticker_history = ticker_history

        except Exception as e:
            print(f"[Memory] LTM fetch error: {e}")

    async def _fetch_rag(
        self,
        context: MemoryContext,
        query: str,
        classification: ClassificationResult,
        budget: TokenBudget
    ) -> None:
        """Fetch from Qdrant RAG (semantic search)."""
        if not self.qdrant or budget.rag_results == 0:
            return

        try:
            from rag.embeddings import embed_texts, sparse_from_text
            from qdrant_client.http import models as rest

            # Embed query
            dense_vecs = await embed_texts([query])
            dense = dense_vecs[0]
            sparse = sparse_from_text(query)

            # Build filters based on classification
            must_conditions = []

            # Filter by ticker if available
            if classification.tickers:
                must_conditions.append(
                    rest.FieldCondition(
                        key="symbol",
                        match=rest.MatchAny(any=classification.tickers)
                    )
                )

            # Limit search based on intent
            limit = 5 if budget.rag_results < 1500 else 8

            # Execute search
            results = self.qdrant.hybrid_search(
                dense=dense,
                sparse=sparse,
                limit=limit,
                must=must_conditions if must_conditions else None
            )

            # Extract chunks
            chunks = []
            for hit in results:
                payload = hit.payload or {}
                chunks.append({
                    "text": payload.get("text", ""),
                    "symbol": payload.get("symbol", ""),
                    "type": payload.get("type", ""),
                    "score": hit.score,
                })

            context.rag_chunks = chunks

        except Exception as e:
            print(f"[Memory] RAG fetch error: {e}")

    # === Store Operations ===

    async def store_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store a conversation message (STM + optionally LTM)."""
        # Always store in STM
        self.stm.add_to_history(session_id, role, content, metadata)

        # Optionally persist important messages to LTM
        # (This could be based on message importance/length)
        return True

    async def store_decision(
        self,
        user_id: str,
        ticker: str,
        query: str,
        decision: Dict[str, Any],
        run_id: Optional[str] = None,
        sentiment: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Store a trading decision.
        Updates LTM and invalidates relevant caches.
        """
        # Store with summaries (updates both tables)
        success = self.summaries.save_decision_with_summaries(
            user_id=user_id,
            ticker=ticker,
            query=query,
            decision=decision,
            sentiment=sentiment,
            **kwargs
        )

        # Invalidate user snapshot cache (version changed)
        if success:
            self.stm.invalidate_user_snapshot(user_id)

        return success

    async def cache_tool_result(
        self,
        run_id: str,
        tool: str,
        result: Any,
        ticker: Optional[str] = None,
        params: Optional[Dict] = None
    ) -> bool:
        """Cache a tool result for the current run."""
        return self.cache.set(run_id, tool, result, ticker, params)

    # === Utility Methods ===

    def get_cached_tool(
        self,
        run_id: str,
        tool: str,
        ticker: Optional[str] = None,
        params: Optional[Dict] = None
    ) -> Optional[Any]:
        """Check if tool result is cached."""
        return self.cache.get(run_id, tool, ticker, params)

    async def clear_session(self, session_id: str) -> bool:
        """Clear all session data."""
        self.stm.clear_history(session_id)
        self.stm.delete(f"session:{session_id}")
        self.stm.delete(f"context_summary:{session_id}")
        return True


# Singleton
_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create MemoryManager singleton."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
