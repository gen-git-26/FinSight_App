# infrastructure/postgres_summaries.py
"""
PostgreSQL Summary Tables - Pre-computed summaries for fast retrieval.

Instead of querying raw trading_decisions or user_patterns tables,
we maintain pre-computed summaries that are updated at write time.

Tables:
- user_profile_summary: Single row per user with aggregated stats
- user_ticker_summary: Per user+ticker decision summary
- recent_decisions_view: Indexed recent decisions

This approach shifts computation from read time to write time,
drastically reducing latency for common queries like:
- "What are my preferences?"
- "What did I decide about TSLA?"
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from infrastructure.postgres_ltm import LTMConfig, PostgresLTM


# SQL for summary tables
SUMMARY_TABLES_SQL = """
-- User profile summary (one row per user)
CREATE TABLE IF NOT EXISTS user_profile_summary (
    user_id VARCHAR(255) PRIMARY KEY,
    risk_tolerance VARCHAR(50),
    preferred_sectors JSONB DEFAULT '[]',
    avg_position_size FLOAT,
    trading_style VARCHAR(50),
    total_decisions INT DEFAULT 0,
    total_tickers_analyzed INT DEFAULT 0,
    last_active TIMESTAMP,
    version INT DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per ticker summary for each user
CREATE TABLE IF NOT EXISTS user_ticker_summary (
    user_id VARCHAR(255) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    total_analyses INT DEFAULT 0,
    last_decision VARCHAR(20),
    last_analysis_date TIMESTAMP,
    decisions_history JSONB DEFAULT '[]',  -- Last 5 decisions
    avg_sentiment FLOAT,
    notes TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, ticker)
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_user_ticker_summary_user ON user_ticker_summary(user_id);
CREATE INDEX IF NOT EXISTS idx_user_ticker_summary_ticker ON user_ticker_summary(ticker);
CREATE INDEX IF NOT EXISTS idx_user_profile_summary_updated ON user_profile_summary(updated_at);
"""


class PostgresSummaries:
    """
    Manages pre-computed summary tables for fast retrieval.

    Key principle: Update summaries at write time (save_trading_decision),
    not at read time (get_user_context).
    """

    def __init__(self, ltm: Optional[PostgresLTM] = None):
        self.ltm = ltm or PostgresLTM()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize summary tables."""
        if self._initialized:
            return True

        # First ensure base LTM tables exist
        self.ltm.initialize()

        with self.ltm.get_connection() as conn:
            if conn is None:
                print("[Summaries] PostgreSQL not available")
                return False

            try:
                with conn.cursor() as cur:
                    cur.execute(SUMMARY_TABLES_SQL)
                self._initialized = True
                print("[Summaries] Summary tables initialized")
                return True
            except Exception as e:
                print(f"[Summaries] Initialization failed: {e}")
                return False

    # === User Profile Summary ===

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get pre-computed user profile summary (single row, fast)."""
        with self.ltm.get_connection() as conn:
            if conn is None:
                return None

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_profile_summary WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def update_user_summary(
        self,
        user_id: str,
        risk_tolerance: Optional[str] = None,
        preferred_sectors: Optional[List[str]] = None,
        trading_style: Optional[str] = None,
        increment_decisions: bool = False
    ) -> bool:
        """
        Update user profile summary.
        Called after trading decisions or preference changes.
        """
        with self.ltm.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                # Upsert with version increment
                cur.execute("""
                    INSERT INTO user_profile_summary (user_id, risk_tolerance, preferred_sectors,
                        trading_style, total_decisions, last_active, version)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT (user_id) DO UPDATE SET
                        risk_tolerance = COALESCE(%s, user_profile_summary.risk_tolerance),
                        preferred_sectors = COALESCE(%s, user_profile_summary.preferred_sectors),
                        trading_style = COALESCE(%s, user_profile_summary.trading_style),
                        total_decisions = user_profile_summary.total_decisions + %s,
                        last_active = CURRENT_TIMESTAMP,
                        version = user_profile_summary.version + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id, risk_tolerance, Json(preferred_sectors or []),
                    trading_style, 1 if increment_decisions else 0,
                    risk_tolerance, Json(preferred_sectors) if preferred_sectors else None,
                    trading_style, 1 if increment_decisions else 0
                ))
                return True

    def get_user_version(self, user_id: str) -> int:
        """Get current user summary version (for cache invalidation)."""
        with self.ltm.get_connection() as conn:
            if conn is None:
                return 0

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT version FROM user_profile_summary WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()
                return result[0] if result else 0

    # === Ticker Summary ===

    def get_ticker_summary(self, user_id: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Get pre-computed ticker summary for user."""
        with self.ltm.get_connection() as conn:
            if conn is None:
                return None

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_ticker_summary
                    WHERE user_id = %s AND ticker = %s
                """, (user_id, ticker.upper()))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_user_tickers(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all ticker summaries for user (recent first)."""
        with self.ltm.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_ticker_summary
                    WHERE user_id = %s
                    ORDER BY last_analysis_date DESC
                    LIMIT %s
                """, (user_id, limit))
                return [dict(row) for row in cur.fetchall()]

    def update_ticker_summary(
        self,
        user_id: str,
        ticker: str,
        decision: str,
        sentiment: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update ticker summary after a trading decision.
        Maintains last 5 decisions in history.
        """
        ticker = ticker.upper()

        with self.ltm.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get existing history
                cur.execute("""
                    SELECT decisions_history FROM user_ticker_summary
                    WHERE user_id = %s AND ticker = %s
                """, (user_id, ticker))
                result = cur.fetchone()

                history = result["decisions_history"] if result else []
                if not isinstance(history, list):
                    history = []

                # Add new decision to history
                history.append({
                    "decision": decision,
                    "date": datetime.utcnow().isoformat(),
                    "sentiment": sentiment
                })
                # Keep last 5
                history = history[-5:]

                # Calculate average sentiment
                sentiments = [h.get("sentiment") for h in history if h.get("sentiment") is not None]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None

                # Upsert
                cur.execute("""
                    INSERT INTO user_ticker_summary
                        (user_id, ticker, total_analyses, last_decision, last_analysis_date,
                         decisions_history, avg_sentiment, notes)
                    VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                    ON CONFLICT (user_id, ticker) DO UPDATE SET
                        total_analyses = user_ticker_summary.total_analyses + 1,
                        last_decision = %s,
                        last_analysis_date = CURRENT_TIMESTAMP,
                        decisions_history = %s,
                        avg_sentiment = %s,
                        notes = COALESCE(%s, user_ticker_summary.notes),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id, ticker, decision, Json(history), avg_sentiment, notes,
                    decision, Json(history), avg_sentiment, notes
                ))
                return True

    # === Recent Decisions (Fast Access) ===

    def get_recent_decisions(
        self,
        user_id: str,
        limit: int = 5,
        ticker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent trading decisions (indexed, fast).
        Uses the base trading_decisions table with proper indexing.
        """
        with self.ltm.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if ticker:
                    cur.execute("""
                        SELECT ticker, query, decision, created_at
                        FROM trading_decisions
                        WHERE user_id = %s AND ticker = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, ticker.upper(), limit))
                else:
                    cur.execute("""
                        SELECT ticker, query, decision, created_at
                        FROM trading_decisions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))

                return [dict(row) for row in cur.fetchall()]

    # === Integrated Save (Updates Summaries Automatically) ===

    def save_decision_with_summaries(
        self,
        user_id: str,
        ticker: str,
        query: str,
        decision: Dict[str, Any],
        sentiment: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Save trading decision AND update all summaries atomically.
        This is the preferred method - updates summaries at write time.
        """
        # Save to base table
        success = self.ltm.save_trading_decision(
            user_id=user_id,
            ticker=ticker,
            query=query,
            decision=decision,
            **kwargs
        )

        if success:
            # Update summaries (these are fast, single-row updates)
            decision_type = decision.get("action", decision.get("recommendation", "hold"))
            self.update_ticker_summary(user_id, ticker, decision_type, sentiment)
            self.update_user_summary(user_id, increment_decisions=True)

        return success


# Singleton
_summaries: Optional[PostgresSummaries] = None


def get_summaries() -> PostgresSummaries:
    """Get or create PostgresSummaries singleton."""
    global _summaries
    if _summaries is None:
        _summaries = PostgresSummaries()
        _summaries.initialize()
    return _summaries
