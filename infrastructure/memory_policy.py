# infrastructure/memory_policy.py
"""
MemoryPolicy — behavioral contract for memory retrieval per query intent.

Maps QueryIntent → which validity classes may influence the answer,
whether live tools are required, and which classes are context-only
(may be shown but never treated as authoritative).

The fundamental rule: market truth always comes from tools, not memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from infrastructure.memory_types import QueryIntent


@dataclass
class MemoryPolicy:
    """Policy governing memory retrieval for a given query intent."""
    intent: str
    allowed_classes: List[str]          # validity classes retrieval may use
    require_live_tools: bool            # must DataFetcher run before composing?
    context_only_classes: List[str]     # allowed but never authoritative
    max_items_per_class: int = 5        # cap on injected memory items
    recency_weight: bool = True         # order by as_of DESC within window


# Intent → MemoryPolicy mapping
# Based on: docs/plans/2026-03-02-ltm-freshness-design.md Section 4
_POLICY_TABLE: dict = {
    QueryIntent.PRICE_ONLY: MemoryPolicy(
        intent="current_price",
        allowed_classes=["user_preference", "behavioral_pattern", "trading_decision"],
        require_live_tools=True,
        context_only_classes=["trading_decision"],
    ),
    QueryIntent.NEWS_SUMMARY: MemoryPolicy(
        intent="latest_news",
        allowed_classes=["news_sentiment", "user_preference"],
        require_live_tools=True,
        context_only_classes=["news_sentiment"],
    ),
    QueryIntent.TRADE_DECISION: MemoryPolicy(
        intent="strategy_recommendation",
        allowed_classes=[
            "trading_decision", "behavioral_pattern", "user_preference",
            "fundamental_data", "news_sentiment",
        ],
        require_live_tools=True,
        context_only_classes=["price_snapshot", "news_sentiment"],
    ),
    QueryIntent.USER_HISTORY: MemoryPolicy(
        intent="explain_last_decision",
        allowed_classes=["trading_decision", "session_memory", "session_summary"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.USER_PREFERENCES: MemoryPolicy(
        intent="preference_lookup",
        allowed_classes=["user_preference", "behavioral_pattern"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.TICKER_INFO: MemoryPolicy(
        intent="company_fundamentals",
        allowed_classes=["fundamental_data", "trading_decision", "user_preference"],
        require_live_tools=True,
        context_only_classes=["fundamental_data"],
    ),
    QueryIntent.SEMANTIC_SEARCH: MemoryPolicy(
        intent="research_compare",
        allowed_classes=["fundamental_data", "news_sentiment", "user_preference", "behavioral_pattern"],
        require_live_tools=True,
        context_only_classes=["fundamental_data", "news_sentiment"],
    ),
    QueryIntent.CONVERSATION: MemoryPolicy(
        intent="recap_conversation",
        allowed_classes=["session_memory", "session_summary"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.UNKNOWN: MemoryPolicy(
        intent="unknown",
        allowed_classes=["user_preference", "session_memory"],
        require_live_tools=True,
        context_only_classes=["session_memory"],
    ),
}


def get_policy(intent: QueryIntent) -> MemoryPolicy:
    """Return the MemoryPolicy for a given query intent."""
    return _POLICY_TABLE.get(intent, _POLICY_TABLE[QueryIntent.UNKNOWN])
