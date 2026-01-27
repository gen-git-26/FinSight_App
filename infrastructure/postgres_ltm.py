# infrastructure/postgres_ltm.py
"""
PostgreSQL Long-Term Memory (LTM) - Persistent storage for user data and history.

Use cases:
- User profiles and preferences
- Trading history and decisions
- Long-term conversation memory
- Analytics and audit logs
"""
from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


@dataclass
class LTMConfig:
    """PostgreSQL LTM configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "finsight"
    user: str = "postgres"
    password: str = ""
    schema: str = "public"

    @classmethod
    def from_env(cls) -> "LTMConfig":
        """Load config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "finsight"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            schema=os.getenv("POSTGRES_SCHEMA", "public")
        )

    @property
    def connection_string(self) -> str:
        """Get connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# SQL for table creation
INIT_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    profile JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading decisions history
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    ticker VARCHAR(20) NOT NULL,
    query TEXT,
    decision JSONB NOT NULL,
    analyst_reports JSONB,
    research_report JSONB,
    risk_assessment JSONB,
    fund_manager_decision JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation memory
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences / learned patterns
CREATE TABLE IF NOT EXISTS user_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trading_decisions_user ON trading_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_ticker ON trading_decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_conversation_memory_user ON conversation_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_patterns_user ON user_patterns(user_id);
"""


class PostgresLTM:
    """
    PostgreSQL-based Long-Term Memory.

    Provides persistent storage for:
    - User profiles and preferences
    - Trading decision history
    - Conversation memory
    - Learned user patterns
    """

    def __init__(self, config: Optional[LTMConfig] = None):
        self.config = config or LTMConfig.from_env()
        self._connection = None
        self._initialized = False

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        if not POSTGRES_AVAILABLE:
            yield None
            return

        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            yield conn
            conn.commit()
        except Exception as e:
            print(f"[LTM] Database error: {e}")
            if conn:
                conn.rollback()
            yield None
        finally:
            if conn:
                conn.close()

    def initialize(self) -> bool:
        """Initialize database tables."""
        if self._initialized:
            return True

        with self.get_connection() as conn:
            if conn is None:
                print("[LTM] PostgreSQL not available")
                return False

            try:
                with conn.cursor() as cur:
                    cur.execute(INIT_SQL)
                self._initialized = True
                print("[LTM] Database initialized")
                return True
            except Exception as e:
                print(f"[LTM] Initialization failed: {e}")
                return False

    # === User Management ===

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile."""
        with self.get_connection() as conn:
            if conn is None:
                return None

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM users WHERE user_id = %s",
                    (user_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None

    def create_or_update_user(
        self,
        user_id: str,
        profile: Optional[Dict] = None,
        preferences: Optional[Dict] = None
    ) -> bool:
        """Create or update user."""
        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (user_id, profile, preferences)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        profile = COALESCE(%s, users.profile),
                        preferences = COALESCE(%s, users.preferences),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id,
                    Json(profile or {}),
                    Json(preferences or {}),
                    Json(profile) if profile else None,
                    Json(preferences) if preferences else None
                ))
                return True

    def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences."""
        return self.create_or_update_user(user_id, preferences=preferences)

    # === Trading Decisions ===

    def save_trading_decision(
        self,
        user_id: str,
        ticker: str,
        query: str,
        decision: Dict[str, Any],
        analyst_reports: Optional[List] = None,
        research_report: Optional[Dict] = None,
        risk_assessment: Optional[Dict] = None,
        fund_manager_decision: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Save a trading decision to history."""
        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_decisions
                    (user_id, session_id, ticker, query, decision,
                     analyst_reports, research_report, risk_assessment, fund_manager_decision)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, session_id, ticker, query,
                    Json(decision),
                    Json(analyst_reports) if analyst_reports else None,
                    Json(research_report) if research_report else None,
                    Json(risk_assessment) if risk_assessment else None,
                    Json(fund_manager_decision) if fund_manager_decision else None
                ))
                return True

    def get_trading_history(
        self,
        user_id: str,
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get trading decision history."""
        with self.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if ticker:
                    cur.execute("""
                        SELECT * FROM trading_decisions
                        WHERE user_id = %s AND ticker = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, ticker, limit))
                else:
                    cur.execute("""
                        SELECT * FROM trading_decisions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))

                return [dict(row) for row in cur.fetchall()]

    # === Conversation Memory ===

    def save_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save a conversation message."""
        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_memory
                    (user_id, session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, session_id, role, content, Json(metadata or {})))
                return True

    def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get conversation history."""
        with self.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if session_id:
                    cur.execute("""
                        SELECT * FROM conversation_memory
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, (user_id, session_id, limit))
                else:
                    cur.execute("""
                        SELECT * FROM conversation_memory
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))

                return [dict(row) for row in cur.fetchall()]

    # === User Patterns ===

    def save_pattern(
        self,
        user_id: str,
        pattern_type: str,
        pattern_data: Dict,
        confidence: float = 0.5
    ) -> bool:
        """Save a learned user pattern."""
        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_patterns
                    (user_id, pattern_type, pattern_data, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, pattern_type)
                    DO UPDATE SET
                        pattern_data = %s,
                        confidence = %s,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id, pattern_type, Json(pattern_data), confidence,
                    Json(pattern_data), confidence
                ))
                return True

    def get_patterns(self, user_id: str) -> List[Dict]:
        """Get user patterns."""
        with self.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_patterns
                    WHERE user_id = %s
                    ORDER BY confidence DESC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]


# Singleton instance
_ltm: Optional[PostgresLTM] = None


def get_ltm() -> PostgresLTM:
    """Get or create LTM singleton."""
    global _ltm
    if _ltm is None:
        _ltm = PostgresLTM()
        _ltm.initialize()
    return _ltm
