# agent/nodes/router.py
"""
Router Agent - Analyzes query and decides which agent should handle it.

Supports A2A (Agent-to-Agent) architecture:
- Trading queries → TradingAgents flow (analysts, researchers, risk, trader)
- Info queries → Standard flow (fetcher, analyst, composer)
- Crypto queries → Crypto agent flow
- General queries → Direct to composer
"""
from __future__ import annotations

import json
import uuid
import httpx
from typing import Dict, Any

from agent.state import AgentState, ParsedQuery
from utils.config import load_settings
from infrastructure.memory_manager import get_memory_manager
from infrastructure.memory_policy import get_policy
from evaluation.metrics import track_metrics


# Trading-related keywords for A2A routing
TRADING_KEYWORDS = [
    'should i buy', 'should i sell', 'trade', 'trading',
    'buy or sell', 'invest', 'investment', 'position',
    'entry', 'exit', 'long', 'short', 'bullish', 'bearish',
    'recommendation', 'analysis', 'analyze', 'forecast',
    'target price', 'price target', 'outlook', 'prediction',
    'האם לקנות', 'האם למכור', 'המלצה', 'ניתוח', 'תחזית'
]

# Trading subtype keywords for granular routing
FUNDAMENTAL_KEYWORDS = [
    'p/e', 'pe ratio', 'eps', 'earnings', 'revenue', 'profit',
    'valuation', 'overvalued', 'undervalued', 'fundamentals',
    'balance sheet', 'income statement', 'cash flow', 'debt',
    'margin', 'growth rate', 'book value', 'dividend'
]

TECHNICAL_KEYWORDS = [
    'rsi', 'macd', 'oversold', 'overbought', 'support', 'resistance',
    'moving average', 'sma', 'ema', 'bollinger', 'volume',
    'trend', 'breakout', 'chart', 'technical', 'indicator'
]

SENTIMENT_KEYWORDS = [
    'sentiment', 'mood', 'feeling', 'outlook', 'opinion',
    'bullish sentiment', 'bearish sentiment', 'fear', 'greed'
]

NEWS_KEYWORDS = [
    'news', 'headline', 'announcement', 'press release',
    'latest', 'recent', 'update', 'breaking'
]


def classify_trading_subtype(query: str) -> str:
    """
    Classify a trading query into a subtype for granular routing.

    Returns one of: full_trading, fundamental, technical, sentiment, news
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in FUNDAMENTAL_KEYWORDS):
        return "fundamental"

    if any(kw in query_lower for kw in TECHNICAL_KEYWORDS):
        return "technical"

    if any(kw in query_lower for kw in NEWS_KEYWORDS):
        return "news"

    if any(kw in query_lower for kw in SENTIMENT_KEYWORDS):
        return "sentiment"

    return "full_trading"


ROUTER_PROMPT = """You are a financial query router. Analyze the query and determine:
1. The type of query (stock, crypto, options, news, fundamentals, comparison, trading, general)
2. Extract ticker symbols
3. Identify intent

**TRADING DETECTION (A2A - routes to TradingAgents):**
- Keywords: should I buy, should I sell, trade, invest, recommendation, analysis, forecast
- Questions about buy/sell decisions → query_type: "trading"
- Requests for trading analysis → query_type: "trading"

**CRYPTO DETECTION:**
- Keywords: bitcoin, btc, ethereum, eth, crypto, solana, dogecoin, ripple, xrp, etc.
- Any cryptocurrency mention → query_type: "crypto"
- Add -USD suffix for crypto tickers: bitcoin → BTC-USD

**STOCK DETECTION:**
- Company names: Tesla, Apple, Microsoft, etc.
- Stock symbols: AAPL, MSFT, TSLA, etc.
- Simple price/info queries → query_type: "stock"

**COMPARISON:**
- Multiple tickers mentioned → query_type: "comparison"
- Keywords: compare, vs, versus, against

Return ONLY this JSON:
{
  "ticker": "AAPL",
  "additional_tickers": [],
  "intent": "price",
  "query_type": "stock",
  "next_agent": "fetcher",
  "is_trading_query": false
}

**next_agent options:**
- "trading" - for trading decisions/analysis (A2A to TradingAgents)
- "crypto" - for crypto queries
- "fetcher" - for simple stock/options/fundamentals info
- "composer" - for general questions (no data fetch needed)

Examples:
"Should I buy Tesla?" → {"ticker": "TSLA", "additional_tickers": [], "intent": "trading", "query_type": "trading", "next_agent": "trading", "is_trading_query": true}
"Analyze AAPL for investment" → {"ticker": "AAPL", "additional_tickers": [], "intent": "analysis", "query_type": "trading", "next_agent": "trading", "is_trading_query": true}
"Bitcoin price" → {"ticker": "BTC-USD", "additional_tickers": [], "intent": "price", "query_type": "crypto", "next_agent": "crypto", "is_trading_query": false}
"Tesla stock price" → {"ticker": "TSLA", "additional_tickers": [], "intent": "price", "query_type": "stock", "next_agent": "fetcher", "is_trading_query": false}
"What is a P/E ratio?" → {"ticker": null, "additional_tickers": [], "intent": "info", "query_type": "general", "next_agent": "composer", "is_trading_query": false}
"""


@track_metrics("router")
async def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node - analyzes the query and decides routing.

    This is the entry point of the graph.
    """
    query = state.get("query", "")
    user_id = state.get("user_id", "default")

    print(f"\n{'='*60}")
    print(f"[Router] Processing: {query}")

    cfg = load_settings()

    # Generate run_id to scope RunCache for this entire query run
    run_id = str(uuid.uuid4())

    # Get enriched memory context via MemoryManager (degrades gracefully if Redis/Qdrant absent)
    manager = get_memory_manager()
    context = None
    memory_context_str = ""
    try:
        context = await manager.get_context(
            query=query,
            session_id=user_id,
            user_id=user_id,
            run_id=run_id,
        )
        memory_context_str = context.to_prompt_context()
    except Exception as e:
        print(f"[Router] Memory fetch failed: {e}")

    # Derive MemoryPolicy from classifier output
    memory_policy = None
    if context and context.classification:
        memory_policy = get_policy(context.classification.intent)

    # Call LLM for routing decision
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": cfg.openai_model,
                "messages": [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": f"Query: {query}\n\nMemory context: {memory_context_str[:500] if memory_context_str else 'None'}"}
                ],
                "temperature": 0.0,
                "max_tokens": 200
            },
            timeout=15.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Clean markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)

        parsed_query = ParsedQuery(
            ticker=parsed.get("ticker"),
            additional_tickers=parsed.get("additional_tickers", []),
            intent=parsed.get("intent", "info"),
            query_type=parsed.get("query_type", "general"),
            raw_query=query
        )

        next_agent = parsed.get("next_agent", "fetcher")
        is_trading_query = parsed.get("is_trading_query", False)

        # A2A: Override to trading if trading keywords detected but LLM missed it
        if not is_trading_query:
            query_lower = query.lower()
            if any(kw in query_lower for kw in TRADING_KEYWORDS):
                is_trading_query = True
                next_agent = "trading"
                parsed_query.query_type = "trading"

        print(f"[Router] → Type: {parsed_query.query_type}, Ticker: {parsed_query.ticker}, Next: {next_agent}")
        print(f"[Router] → A2A Trading Mode: {is_trading_query}")

        return {
            "parsed_query": parsed_query,
            "next_agent": next_agent,
            "is_trading_query": is_trading_query,
            "run_id": run_id,
            "memory_context": context,
            "memory_policy": memory_policy,
            "memory": {
                "user_id": user_id,
                "retrieved_memory": memory_context_str,
            }
        }

    except Exception as e:
        print(f"[Router] Error: {e}")
        # Fallback routing
        query_lower = query.lower()

        is_crypto = any(kw in query_lower for kw in [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto',
            'solana', 'sol', 'dogecoin', 'doge', 'xrp', 'ripple'
        ])

        is_trading = any(kw in query_lower for kw in TRADING_KEYWORDS)

        if is_trading:
            next_agent = "trading"
            query_type = "trading"
        elif is_crypto:
            next_agent = "crypto"
            query_type = "crypto"
        else:
            next_agent = "fetcher"
            query_type = "general"

        return {
            "parsed_query": ParsedQuery(
                ticker=None,
                intent="info",
                query_type=query_type,
                raw_query=query
            ),
            "next_agent": next_agent,
            "is_trading_query": is_trading,
            "run_id": run_id,
            "memory_context": context,
            "memory_policy": memory_policy,
            "error": str(e)
        }
