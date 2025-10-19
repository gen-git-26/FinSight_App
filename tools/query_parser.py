# tools/query_parser.py - UPDATED with CryptoResolver support
from __future__ import annotations

import json
import httpx
import asyncio
from typing import Optional, List
from dataclasses import dataclass
from utils.config import load_settings
from tools.async_utils import run_async_safe
from tools.time_parser import parse_time_range_to_days
from tools.crypto_resolver import get_crypto_resolver


@dataclass
class ParsedQuery:
    """Structured representation of a financial query."""
    primary_ticker: Optional[str]
    secondary_tickers: List[str]
    intent: str  # "price_quote", "news", "analysis", "options", etc.
    data_type: str  # "realtime", "historical", "fundamental"
    time_range: Optional[str]  # "30 days", "1 year", etc.
    specific_date: Optional[str]  # YYYY-MM-DD
    options_type: Optional[str]  # "calls", "puts"
    expiration_days: Optional[int]
    company_names: List[str]
    raw_intent: str  # original query
    is_crypto: bool = False  # Is this a crypto query?


async def _parse_query_with_llm_async(query: str) -> ParsedQuery:
    """
    Use OpenAI GPT to parse the user query structurally.
    Enhanced with crypto detection and CryptoResolver.
    """
    cfg = load_settings()
    
    system_prompt = """You are a financial query parser specialized in detecting crypto queries.

Return ONLY valid JSON (no markdown, no code blocks, just raw JSON):
{
  "primary_ticker": "AAPL or BTC or null",
  "secondary_tickers": ["GOOGL"],
  "intent": "one of: price_quote, news, analysis, comparison, options, fundamentals, dividend, insider, recommendation, historical",
  "data_type": "one of: realtime, historical, fundamental",
  "time_range": "30 days or null",
  "specific_date": "YYYY-MM-DD or null",
  "options_type": "calls or puts or chain or null",
  "expiration_days": 30 or null,
  "company_names": ["Apple Inc"],
  "is_crypto": true or false
}

CRITICAL RULES:
1. Crypto Detection:
   - Keywords: bitcoin, ethereum, solana, cardano, dogecoin, ripple, etc.
   - Suffixes: -USD (e.g., BTC-USD)
   - Set is_crypto: true if detected
   
2. Symbol Resolution:
   - "bitcoin" → "BTC"
   - "ethereum" → "ETH"
   - "solana" → "SOL"
   - "dogecoin" → "DOGE"
   - "cardano" → "ADA"
   - "ripple" → "XRP"
   
3. If is_crypto=true:
   - Return uppercase symbol
   - CryptoResolver will handle -USD suffix
   
4. Intent Detection:
   - "price", "quote", "what is X worth" → price_quote
   - "news", "latest" → news
   - "analyze", "analysis" → analysis
   - "compare" → comparison
   - "options", "calls", "puts" → options
   - "fundamentals", "financial" → fundamentals
   - "history", "historical" → historical

Examples:
- "solana price" → {"primary_ticker": "SOL", "intent": "price_quote", "is_crypto": true}
- "bitcoin news" → {"primary_ticker": "BTC", "intent": "news", "is_crypto": true}
- "Apple stock" → {"primary_ticker": "AAPL", "intent": "price_quote", "is_crypto": false}
- "ethereum vs bitcoin" → {"primary_ticker": "ETH", "secondary_tickers": ["BTC"], "is_crypto": true}
"""

    user_message = f"""Parse this financial query:

Query: {query}

Return ONLY valid JSON, no other text."""

    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": cfg.openai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            
            content = data["choices"][0]["message"]["content"].strip()
            
            # Extract JSON from various formats
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            parsed = json.loads(content)
            
            # Post-process ticker with CryptoResolver
            primary_ticker = parsed.get("primary_ticker")
            is_crypto = parsed.get("is_crypto", False)
            
            # If marked as crypto, verify with resolver
            if is_crypto and primary_ticker:
                crypto_resolver = get_crypto_resolver()
                crypto_result = crypto_resolver.resolve(primary_ticker, use_cache_only=True)
                if crypto_result['found']:
                    primary_ticker = crypto_result['symbol']
                    print(f"[query_parser] Resolved crypto: {parsed.get('primary_ticker')} → {primary_ticker}")
            
            return ParsedQuery(
                primary_ticker=primary_ticker,
                secondary_tickers=parsed.get("secondary_tickers", []),
                intent=parsed.get("intent", "analysis"),
                data_type=parsed.get("data_type", "realtime"),
                time_range=parsed.get("time_range"),
                specific_date=parsed.get("specific_date"),
                options_type=parsed.get("options_type"),
                expiration_days=parsed.get("expiration_days"),
                company_names=parsed.get("company_names", []),
                raw_intent=query,
                is_crypto=is_crypto
            )
    
    except json.JSONDecodeError as e:
        print(f"[query_parser] JSON decode error: {e}")
        print(f"[query_parser] Content: {content[:200]}")
        return _default_parsed_query(query)
    
    except Exception as e:
        print(f"[query_parser] Error: {e}")
        return _default_parsed_query(query)


def _default_parsed_query(query: str) -> ParsedQuery:
    """Fallback parsed query."""
    # Try quick crypto detection from keywords
    query_lower = query.lower()
    is_crypto = any(k in query_lower for k in ['bitcoin', 'ethereum', 'solana', 'crypto', 'coin', '-usd'])
    
    return ParsedQuery(
        primary_ticker=None,
        secondary_tickers=[],
        intent="analysis",
        data_type="realtime",
        time_range=None,
        specific_date=None,
        options_type=None,
        expiration_days=None,
        company_names=[],
        raw_intent=query,
        is_crypto=is_crypto
    )


def parse_query_with_llm(query: str) -> ParsedQuery:
    """
    Safe sync wrapper around async parser.
    Can be called from Streamlit or sync context without event loop conflicts.
    """
    try:
        return run_async_safe(_parse_query_with_llm_async(query))
    except Exception as e:
        print(f"[parse_query_with_llm] Wrapper error: {e}")
        return _default_parsed_query(query)