# infrastructure/query_classifier.py
"""
2-Stage Query Classifier for intelligent memory routing.

Stage 1 (Deterministic, <1ms):
- Regex patterns for tickers
- Keyword matching for intents
- Intent verb detection

Stage 2 (LLM Fallback, ~100ms):
- Only if confidence < threshold
- Small/fast model for intent classification
- Returns structured intent + confidence

This approach maximizes hit rate on fast path while
falling back to LLM only when truly ambiguous.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from infrastructure.memory_types import (
    QueryIntent,
    ClassificationResult,
    MemoryLayer,
)


# === Stage 1: Deterministic Patterns ===

# Ticker patterns (US stocks, crypto)
TICKER_PATTERN = re.compile(
    r'\b([A-Z]{1,5})\b'  # 1-5 uppercase letters
    r'|'
    r'\$([A-Za-z]{1,5})'  # $AAPL format
)

# Common crypto tickers
CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "BCH", "ALGO"
}

# Intent patterns with confidence scores
INTENT_PATTERNS: Dict[QueryIntent, List[Tuple[re.Pattern, float]]] = {
    QueryIntent.PRICE_ONLY: [
        (re.compile(r'\b(price|מחיר|quote|שער)\b', re.I), 0.9),
        (re.compile(r'\b(how much|כמה|worth|שווה)\b', re.I), 0.7),
        (re.compile(r'\b(current|נוכחי)\s+(price|מחיר)\b', re.I), 0.95),
    ],
    QueryIntent.TICKER_INFO: [
        (re.compile(r'\b(info|מידע|about|על|details|פרטים)\b', re.I), 0.8),
        (re.compile(r'\b(tell me about|ספר לי על)\b', re.I), 0.85),
        (re.compile(r'\b(what is|מה זה)\s+[A-Z]{1,5}\b', re.I), 0.8),
    ],
    QueryIntent.NEWS_SUMMARY: [
        (re.compile(r'\b(news|חדשות|headlines|כותרות)\b', re.I), 0.9),
        (re.compile(r'\b(latest|אחרונות|recent|עדכניות)\b', re.I), 0.6),
        (re.compile(r'\b(what.+happening|מה קורה)\b', re.I), 0.7),
    ],
    QueryIntent.TRADE_DECISION: [
        (re.compile(r'\b(should i|האם כדאי|buy|sell|hold|לקנות|למכור|להחזיק)\b', re.I), 0.85),
        (re.compile(r'\b(recommend|המלצה|analysis|אנליזה|analyze|לנתח)\b', re.I), 0.8),
        (re.compile(r'\b(invest|להשקיע|position|פוזיציה)\b', re.I), 0.75),
        (re.compile(r'\b(good time|זמן טוב)\b', re.I), 0.7),
    ],
    QueryIntent.USER_HISTORY: [
        (re.compile(r'\b(what did i|מה אמרתי|my previous|הקודם שלי)\b', re.I), 0.9),
        (re.compile(r'\b(earlier|קודם|before|לפני)\b', re.I), 0.6),
        (re.compile(r'\b(history|היסטוריה|past|עבר)\s+(decision|החלטה)\b', re.I), 0.85),
    ],
    QueryIntent.USER_PREFERENCES: [
        (re.compile(r'\b(my preference|העדפות שלי|i prefer|אני מעדיף)\b', re.I), 0.9),
        (re.compile(r'\b(risk tolerance|סיבולת סיכון)\b', re.I), 0.95),
        (re.compile(r'\b(my style|הסגנון שלי|trading style)\b', re.I), 0.85),
        (re.compile(r'\b(what do i like|מה אני אוהב)\b', re.I), 0.8),
    ],
    QueryIntent.SEMANTIC_SEARCH: [
        (re.compile(r'\b(similar|דומה|like|כמו|find|מצא)\b', re.I), 0.7),
        (re.compile(r'\b(related|קשור|compare|השווה)\b', re.I), 0.7),
        (re.compile(r'\b(search|חפש|look for|חפש)\b', re.I), 0.65),
    ],
    QueryIntent.CONVERSATION: [
        (re.compile(r'\b(you said|אמרת|we discussed|דיברנו)\b', re.I), 0.85),
        (re.compile(r'\b(continue|המשך|more|עוד)\b', re.I), 0.6),
        (re.compile(r'\b(explain|הסבר|elaborate|פרט)\b', re.I), 0.65),
    ],
}

# Memory layer mapping by intent
INTENT_TO_LAYERS: Dict[QueryIntent, List[MemoryLayer]] = {
    QueryIntent.PRICE_ONLY: [MemoryLayer.RUN_CACHE],
    QueryIntent.TICKER_INFO: [MemoryLayer.RUN_CACHE, MemoryLayer.RAG],
    QueryIntent.NEWS_SUMMARY: [MemoryLayer.RUN_CACHE, MemoryLayer.RAG],
    QueryIntent.TRADE_DECISION: [MemoryLayer.RUN_CACHE, MemoryLayer.LTM, MemoryLayer.RAG],
    QueryIntent.USER_HISTORY: [MemoryLayer.STM, MemoryLayer.LTM],
    QueryIntent.USER_PREFERENCES: [MemoryLayer.STM, MemoryLayer.LTM],
    QueryIntent.SEMANTIC_SEARCH: [MemoryLayer.RAG],
    QueryIntent.CONVERSATION: [MemoryLayer.STM],
    QueryIntent.UNKNOWN: [MemoryLayer.STM, MemoryLayer.RUN_CACHE],  # Conservative default
}

# Confidence threshold for LLM fallback
CONFIDENCE_THRESHOLD = 0.65


class QueryClassifier:
    """
    2-Stage query classifier for memory routing.

    Stage 1: Fast deterministic classification (~0.1ms)
    Stage 2: LLM fallback for low confidence cases (~100ms)
    """

    def __init__(self, llm_fallback: bool = True):
        """
        Args:
            llm_fallback: Whether to use LLM for low-confidence cases
        """
        self.llm_fallback = llm_fallback
        self._llm = None

    # === Stage 1: Deterministic Classification ===

    def _extract_tickers(self, query: str) -> List[str]:
        """Extract ticker symbols from query."""
        tickers = set()

        for match in TICKER_PATTERN.finditer(query):
            ticker = match.group(1) or match.group(2)
            if ticker:
                ticker = ticker.upper()
                # Filter out common words
                if ticker not in {"I", "A", "THE", "AND", "OR", "IS", "IT", "TO", "IN", "ON", "FOR"}:
                    # Check if it's a known crypto or looks like a stock ticker
                    if ticker in CRYPTO_TICKERS or len(ticker) >= 2:
                        tickers.add(ticker)

        return list(tickers)

    def _classify_deterministic(self, query: str) -> Tuple[QueryIntent, float, List[str]]:
        """
        Stage 1: Fast deterministic classification.
        Returns (intent, confidence, keywords).
        """
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        matched_keywords = []

        # Check each intent pattern
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern, base_confidence in patterns:
                match = pattern.search(query)
                if match:
                    # Boost confidence if multiple patterns match
                    confidence = base_confidence
                    matched_keywords.append(match.group(0))

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent

        # Boost confidence if tickers found for certain intents
        tickers = self._extract_tickers(query)
        if tickers and best_intent in {
            QueryIntent.PRICE_ONLY, QueryIntent.TICKER_INFO,
            QueryIntent.NEWS_SUMMARY, QueryIntent.TRADE_DECISION
        }:
            best_confidence = min(best_confidence + 0.1, 1.0)

        return best_intent, best_confidence, matched_keywords

    # === Stage 2: LLM Fallback ===

    async def _classify_with_llm(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Stage 2: LLM-based classification for ambiguous queries.
        Uses a small, fast model.
        """
        if not self.llm_fallback:
            return QueryIntent.UNKNOWN, 0.5

        try:
            from openai import AsyncOpenAI
            import os

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Short, focused prompt
            prompt = f"""Classify this query's intent. Return ONLY the intent name and confidence (0-1).

Intents: PRICE_ONLY, TICKER_INFO, NEWS_SUMMARY, TRADE_DECISION, USER_HISTORY, USER_PREFERENCES, SEMANTIC_SEARCH, CONVERSATION

Query: "{query}"

Response format: INTENT_NAME 0.X"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, cheap model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )

            result = response.choices[0].message.content.strip()
            parts = result.split()

            if len(parts) >= 2:
                intent_name = parts[0].upper()
                confidence = float(parts[1])

                # Map to enum
                intent_map = {
                    "PRICE_ONLY": QueryIntent.PRICE_ONLY,
                    "TICKER_INFO": QueryIntent.TICKER_INFO,
                    "NEWS_SUMMARY": QueryIntent.NEWS_SUMMARY,
                    "TRADE_DECISION": QueryIntent.TRADE_DECISION,
                    "USER_HISTORY": QueryIntent.USER_HISTORY,
                    "USER_PREFERENCES": QueryIntent.USER_PREFERENCES,
                    "SEMANTIC_SEARCH": QueryIntent.SEMANTIC_SEARCH,
                    "CONVERSATION": QueryIntent.CONVERSATION,
                }

                return intent_map.get(intent_name, QueryIntent.UNKNOWN), confidence

        except Exception as e:
            print(f"[Classifier] LLM fallback failed: {e}")

        return QueryIntent.UNKNOWN, 0.5

    # === Main Classification ===

    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify query intent and determine memory layers needed.

        Returns ClassificationResult with:
        - intent: Detected query intent
        - confidence: 0.0-1.0 confidence score
        - tickers: Extracted ticker symbols
        - layers_needed: Which memory layers to query
        """
        # Stage 1: Deterministic
        intent, confidence, keywords = self._classify_deterministic(query)
        tickers = self._extract_tickers(query)

        # Stage 2: LLM fallback if low confidence
        if confidence < CONFIDENCE_THRESHOLD and self.llm_fallback:
            llm_intent, llm_confidence = await self._classify_with_llm(query)

            # Use LLM result if more confident
            if llm_confidence > confidence:
                intent = llm_intent
                confidence = llm_confidence

        # Determine layers needed
        layers = INTENT_TO_LAYERS.get(intent, INTENT_TO_LAYERS[QueryIntent.UNKNOWN])

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            tickers=tickers,
            keywords=keywords,
            layers_needed=layers
        )

    def classify_sync(self, query: str) -> ClassificationResult:
        """
        Synchronous classification (Stage 1 only, no LLM).
        Use when you need instant results and can tolerate lower accuracy.
        """
        intent, confidence, keywords = self._classify_deterministic(query)
        tickers = self._extract_tickers(query)
        layers = INTENT_TO_LAYERS.get(intent, INTENT_TO_LAYERS[QueryIntent.UNKNOWN])

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            tickers=tickers,
            keywords=keywords,
            layers_needed=layers
        )


# Singleton
_classifier: Optional[QueryClassifier] = None


def get_classifier(llm_fallback: bool = True) -> QueryClassifier:
    """Get or create QueryClassifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier(llm_fallback=llm_fallback)
    return _classifier
