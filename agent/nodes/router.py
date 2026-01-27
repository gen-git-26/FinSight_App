# agent/nodes/router.py
"""
Router Agent - Analyzes query and decides which agent should handle it.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any

from agent.state import AgentState, ParsedQuery
from utils.config import load_settings
from memory.manager import fetch_memory, _session


ROUTER_PROMPT = """You are a financial query router. Analyze the query and determine:
1. The type of query (stock, crypto, options, news, fundamentals, comparison, general)
2. Extract ticker symbols
3. Identify intent

**CRYPTO DETECTION:**
- Keywords: bitcoin, btc, ethereum, eth, crypto, solana, dogecoin, ripple, xrp, etc.
- Any cryptocurrency mention → query_type: "crypto"
- Add -USD suffix for crypto tickers: bitcoin → BTC-USD

**STOCK DETECTION:**
- Company names: Tesla, Apple, Microsoft, etc.
- Stock symbols: AAPL, MSFT, TSLA, etc.
- query_type: "stock" or "options" or "fundamentals" based on intent

**COMPARISON:**
- Multiple tickers mentioned → query_type: "comparison"
- Keywords: compare, vs, versus, against

Return ONLY this JSON:
{
  "ticker": "AAPL",
  "additional_tickers": [],
  "intent": "price",
  "query_type": "stock",
  "next_agent": "fetcher"
}

**next_agent options:**
- "crypto" - for crypto queries
- "fetcher" - for stock/options/fundamentals
- "composer" - for general questions (no data fetch needed)

Examples:
"Bitcoin price" → {"ticker": "BTC-USD", "additional_tickers": [], "intent": "price", "query_type": "crypto", "next_agent": "crypto"}
"Tesla stock" → {"ticker": "TSLA", "additional_tickers": [], "intent": "price", "query_type": "stock", "next_agent": "fetcher"}
"Compare AAPL and MSFT" → {"ticker": "AAPL", "additional_tickers": ["MSFT"], "intent": "comparison", "query_type": "comparison", "next_agent": "fetcher"}
"What is a P/E ratio?" → {"ticker": null, "additional_tickers": [], "intent": "info", "query_type": "general", "next_agent": "composer"}
"""


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node - analyzes the query and decides routing.

    This is the entry point of the graph.
    """
    query = state.get("query", "")
    user_id = state.get("user_id", "default")

    print(f"\n{'='*60}")
    print(f"[Router] Processing: {query}")

    cfg = load_settings()

    # Get memory context
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                memory_context = pool.submit(
                    asyncio.run, fetch_memory(query, k=3)
                ).result()
        else:
            memory_context = asyncio.run(fetch_memory(query, k=3))
    except Exception as e:
        print(f"[Router] Memory fetch failed: {e}")
        memory_context = ""

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
                    {"role": "user", "content": f"Query: {query}\n\nMemory context: {memory_context[:500] if memory_context else 'None'}"}
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

        print(f"[Router] → Type: {parsed_query.query_type}, Ticker: {parsed_query.ticker}, Next: {next_agent}")

        return {
            "parsed_query": parsed_query,
            "next_agent": next_agent,
            "memory": {
                "user_id": user_id,
                "session_history": _session.context(),
                "retrieved_memory": memory_context
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

        return {
            "parsed_query": ParsedQuery(
                ticker=None,
                intent="info",
                query_type="crypto" if is_crypto else "general",
                raw_query=query
            ),
            "next_agent": "crypto" if is_crypto else "fetcher",
            "error": str(e)
        }
