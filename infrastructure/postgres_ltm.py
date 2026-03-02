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

-- Trading decisions history (permanent audit log — never deleted)
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
    validity_class VARCHAR(50) DEFAULT 'trading_decision',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_valid_after_as_of_td
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- Conversation memory
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    validity_class VARCHAR(50) DEFAULT 'session_memory',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_valid_after_as_of_cm
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- User preferences / learned patterns
CREATE TABLE IF NOT EXISTS user_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    validity_class VARCHAR(50) DEFAULT 'behavioral_pattern',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, pattern_type),
    CONSTRAINT chk_valid_after_as_of_up
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- Validity window trigger function
CREATE OR REPLACE FUNCTION set_validity_window()
RETURNS TRIGGER AS $$
DECLARE
    horizon TEXT;
BEGIN
    -- Respect explicit value if caller set it
    IF NEW.valid_for_context_until IS NOT NULL THEN
        RETURN NEW;
    END IF;

    -- trading_decision: check horizon from decision JSONB
    IF NEW.validity_class = 'trading_decision' THEN
        horizon := NEW.decision->>'horizon';
        NEW.valid_for_context_until := NEW.as_of + (
            CASE horizon
                WHEN 'day'       THEN INTERVAL '7 days'
                WHEN 'long_term' THEN INTERVAL '180 days'
                ELSE                  INTERVAL '30 days'
            END
        );
        RETURN NEW;
    END IF;

    NEW.valid_for_context_until := CASE NEW.validity_class
        WHEN 'price_snapshot'     THEN NEW.as_of + INTERVAL '1 hour'
        WHEN 'end_of_day_price'   THEN NEW.as_of + INTERVAL '48 hours'
        WHEN 'breaking_news'      THEN NEW.as_of + INTERVAL '72 hours'
        WHEN 'news_sentiment'     THEN NEW.as_of + INTERVAL '7 days'
        WHEN 'session_memory'     THEN NEW.as_of + INTERVAL '3 days'
        WHEN 'session_summary'    THEN NEW.as_of + INTERVAL '30 days'
        WHEN 'fundamental_data'   THEN NEW.as_of + INTERVAL '90 days'
        WHEN 'behavioral_pattern' THEN NEW.as_of + INTERVAL '180 days'
        WHEN 'user_preference'    THEN NULL
        ELSE                           NEW.as_of + INTERVAL '30 days'
    END;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_td_validity
    BEFORE INSERT ON trading_decisions
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

CREATE OR REPLACE TRIGGER trg_cm_validity
    BEFORE INSERT ON conversation_memory
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

CREATE OR REPLACE TRIGGER trg_up_validity
    BEFORE INSERT ON user_patterns
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trading_decisions_user ON trading_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_ticker ON trading_decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_conversation_memory_user ON conversation_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_patterns_user ON user_patterns(user_id);

-- Compound indexes for validity-filtered retrieval
CREATE INDEX IF NOT EXISTS idx_td_user_class_valid
    ON trading_decisions (user_id, validity_class, valid_for_context_until DESC);
CREATE INDEX IF NOT EXISTS idx_cm_user_session_valid
    ON conversation_memory (user_id, session_id, valid_for_context_until DESC);
CREATE INDEX IF NOT EXISTS idx_up_user_class_valid
    ON user_patterns (user_id, validity_class, valid_for_context_until DESC);
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
        session_id: Optional[str] = None,
        validity_class: str = "trading_decision",
        as_of: Optional[int] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Save a trading decision to history."""
        as_of_ts = datetime.fromtimestamp(as_of) if as_of else datetime.utcnow()

        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_decisions
                    (user_id, session_id, ticker, query, decision,
                     analyst_reports, research_report, risk_assessment,
                     fund_manager_decision, validity_class, as_of, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, session_id, ticker, query,
                    Json(decision),
                    Json(analyst_reports) if analyst_reports else None,
                    Json(research_report) if research_report else None,
                    Json(risk_assessment) if risk_assessment else None,
                    Json(fund_manager_decision) if fund_manager_decision else None,
                    validity_class, as_of_ts, source
                ))
                return True

    def get_trading_history(
        self,
        user_id: str,
        ticker: Optional[str] = None,
        limit: int = 50,
        context_only: bool = True,
    ) -> List[Dict]:
        """Get trading decision history.

        Args:
            context_only: If True (default), filter out expired records.
                          Set False only for audit/admin queries.
        """
        freshness_clause = (
            "AND (valid_for_context_until > NOW() OR valid_for_context_until IS NULL)"
            if context_only else ""
        )
        with self.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if ticker:
                    cur.execute(f"""
                        SELECT * FROM trading_decisions
                        WHERE user_id = %s AND ticker = %s {freshness_clause}
                        ORDER BY as_of DESC
                        LIMIT %s
                    """, (user_id, ticker, limit))
                else:
                    cur.execute(f"""
                        SELECT * FROM trading_decisions
                        WHERE user_id = %s {freshness_clause}
                        ORDER BY as_of DESC
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
        metadata: Optional[Dict] = None,
        validity_class: str = "session_memory",
        as_of: Optional[int] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Save a conversation message."""
        as_of_ts = datetime.fromtimestamp(as_of) if as_of else datetime.utcnow()

        with self.get_connection() as conn:
            if conn is None:
                return False

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_memory
                    (user_id, session_id, role, content, metadata,
                     validity_class, as_of, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, session_id, role, content, Json(metadata or {}),
                    validity_class, as_of_ts, source
                ))
                return True

    def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
        context_only: bool = True,
    ) -> List[Dict]:
        """Get conversation history."""
        freshness_clause = (
            "AND (valid_for_context_until > NOW() OR valid_for_context_until IS NULL)"
            if context_only else ""
        )
        with self.get_connection() as conn:
            if conn is None:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if session_id:
                    cur.execute(f"""
                        SELECT * FROM conversation_memory
                        WHERE user_id = %s AND session_id = %s {freshness_clause}
                        ORDER BY as_of ASC
                        LIMIT %s
                    """, (user_id, session_id, limit))
                else:
                    cur.execute(f"""
                        SELECT * FROM conversation_memory
                        WHERE user_id = %s {freshness_clause}
                        ORDER BY as_of DESC
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
