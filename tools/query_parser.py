# tools/query_parser.py - NO HARDCODING, JUST SMART LLM
from __future__ import annotations

import json
import httpx
import re
from typing import Optional, List
from dataclasses import dataclass
from utils.config import load_settings


@dataclass
class ParsedQuery:
    """Simple parsed query."""
    ticker: Optional[str]
    additional_tickers: List[str]
    intent: str
    raw_query: str


def parse_query_simple(query: str) -> ParsedQuery:
    """
    Use LLM to extract tickers and intent.
    LLM knows all company names → tickers naturally.
    """
    cfg = load_settings()
    
    system_prompt = """You are a financial query parser. Extract stock tickers and intent.

**CRITICAL**: Always convert company names to their stock tickers using your knowledge.

Return ONLY this JSON (no markdown, no extra text):
{
  "ticker": "AMZN",
  "additional_tickers": ["GOOGL"],
  "intent": "fundamentals"
}

**Rules:**
1. ticker: Main stock symbol
   - Convert names: "Amazon" → "AMZN", "Google"/"Alphabet" → "GOOGL", "Apple" → "AAPL"
   - Keep existing symbols: "AAPL" → "AAPL"
   - Crypto: add -USD suffix: "bitcoin" → "BTC-USD"
   - If no ticker/company found: null

2. additional_tickers: Array of other tickers mentioned (for comparisons)
   - "Amazon and Google" → ticker: "AMZN", additional_tickers: ["GOOGL"]
   - Empty array [] if only one company

3. intent: ONE word
   - "price" | "news" | "options" | "fundamentals" | "historical" | "info"

**Examples:**
"Compare the quarterly income statements of Amazon and Google"
→ {"ticker": "AMZN", "additional_tickers": ["GOOGL"], "intent": "fundamentals"}

"Tesla stock price"
→ {"ticker": "TSLA", "additional_tickers": [], "intent": "price"}

"AAPL vs MSFT"
→ {"ticker": "AAPL", "additional_tickers": ["MSFT"], "intent": "info"}

"Meta Platforms news"
→ {"ticker": "META", "additional_tickers": [], "intent": "news"}

"bitcoin price"
→ {"ticker": "BTC-USD", "additional_tickers": [], "intent": "price"}

"What are the key financial metrics for Tesla from the stock info"
→ {"ticker": "TSLA", "additional_tickers": [], "intent": "fundamentals"}
"""

    try:
        print(f"[query_parser] Calling LLM for: {query[:80]}...")
        
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": cfg.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse: {query}"}
                ],
                "temperature": 0.0,
                "max_tokens": 150
            },
            timeout=20.0  # Longer timeout
        )
        
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        print(f"[query_parser] LLM response: {content[:200]}")
        
        # Clean markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1].strip()
        
        # Parse JSON
        parsed = json.loads(content)
        
        result = ParsedQuery(
            ticker=parsed.get("ticker"),
            additional_tickers=parsed.get("additional_tickers", []),
            intent=parsed.get("intent", "info"),
            raw_query=query
        )
        
        print(f"[query_parser] ✓ Parsed: ticker={result.ticker}, additional={result.additional_tickers}, intent={result.intent}")
        return result
    
    except json.JSONDecodeError as e:
        print(f"[query_parser] ✗ JSON parse error: {e}")
        print(f"[query_parser] Content was: {content[:500]}")
        return _fallback_parse(query)
    
    except httpx.TimeoutException:
        print(f"[query_parser] ✗ LLM timeout")
        return _fallback_parse(query)
    
    except Exception as e:
        print(f"[query_parser] ✗ Error: {e}")
        return _fallback_parse(query)


def _fallback_parse(query: str) -> ParsedQuery:
    """
    Simple fallback if LLM fails.
    Only uses basic regex - no hardcoded mappings.
    """
    print(f"[query_parser] Using fallback parsing")
    
    # Extract explicit tickers (3-5 uppercase letters, optionally -USD)
    ticker_pattern = r'\b([A-Z]{2,5}(?:-USD)?)\b'
    matches = re.findall(ticker_pattern, query.upper())
    
    # Filter out common English words
    noise_words = {'USD', 'GET', 'THE', 'FOR', 'AND', 'OR', 'FROM', 'WITH'}
    tickers = [m for m in matches if m not in noise_words]
    
    # Detect intent from keywords
    q = query.lower()
    if 'option' in q:
        intent = 'options'
    elif 'news' in q:
        intent = 'news'
    elif 'price' in q or 'quote' in q:
        intent = 'price'
    elif 'financial' in q or 'statement' in q or 'fundamental' in q or 'income' in q:
        intent = 'fundamentals'
    elif 'history' in q or 'historical' in q:
        intent = 'historical'
    else:
        intent = 'info'
    
    # Split tickers
    primary = tickers[0] if tickers else None
    additional = tickers[1:] if len(tickers) > 1 else []
    
    print(f"[query_parser] Fallback result: ticker={primary}, additional={additional}, intent={intent}")
    
    return ParsedQuery(
        ticker=primary,
        additional_tickers=additional,
        intent=intent,
        raw_query=query
    )