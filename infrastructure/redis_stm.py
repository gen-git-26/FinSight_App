# infrastructure/redis_stm.py
"""
Redis Short-Term Memory (STM) - Fast session cache and temporary storage.

Use cases:
- Session conversation history
- Agent state caching
- Temporary data between agents
- Rate limiting / deduplication
"""
from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class STMConfig:
    """Redis STM configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    prefix: str = "finsight:"

    @classmethod
    def from_env(cls) -> "STMConfig":
        """Load config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            default_ttl=int(os.getenv("REDIS_TTL", "3600")),
            prefix=os.getenv("REDIS_PREFIX", "finsight:")
        )


class RedisSTM:
    """
    Redis-based Short-Term Memory.

    Provides fast caching for:
    - Session state
    - Conversation history
    - Agent intermediate results
    - Temporary data
    """

    def __init__(self, config: Optional[STMConfig] = None):
        self.config = config or STMConfig.from_env()
        self._client: Optional[redis.Redis] = None
        self._fallback: Dict[str, Any] = {}  # In-memory fallback

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get or create Redis client."""
        if not REDIS_AVAILABLE:
            return None

        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    decode_responses=True
                )
                # Test connection
                self._client.ping()
            except Exception as e:
                print(f"[STM] Redis connection failed: {e}, using in-memory fallback")
                self._client = None

        return self._client

    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.config.prefix}{key}"

    # === Basic Operations ===

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value with optional TTL."""
        full_key = self._key(key)
        ttl = ttl or self.config.default_ttl

        if self.client:
            try:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.client.setex(full_key, ttl, value)
                return True
            except Exception as e:
                print(f"[STM] Set failed: {e}")

        # Fallback
        self._fallback[full_key] = value
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value."""
        full_key = self._key(key)

        if self.client:
            try:
                value = self.client.get(full_key)
                if value is None:
                    return default
                try:
                    return json.loads(value)
                except:
                    return value
            except Exception as e:
                print(f"[STM] Get failed: {e}")

        # Fallback
        return self._fallback.get(full_key, default)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        full_key = self._key(key)

        if self.client:
            try:
                self.client.delete(full_key)
            except:
                pass

        self._fallback.pop(full_key, None)
        return True

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._key(key)

        if self.client:
            try:
                return self.client.exists(full_key) > 0
            except:
                pass

        return full_key in self._fallback

    # === Session Management ===

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data."""
        return self.get(f"session:{session_id}", {})

    def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = 7200) -> bool:
        """Set session data (default 2 hour TTL)."""
        return self.set(f"session:{session_id}", data, ttl)

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data."""
        session = self.get_session(session_id)
        session.update(updates)
        return self.set_session(session_id, session)

    # === Conversation History ===

    def add_to_history(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
        max_turns: int = 25
    ) -> bool:
        """Add a message to conversation history."""
        history_key = f"history:{session_id}"
        history = self.get(history_key, [])

        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        history.append(message)

        # Keep only last N turns
        if len(history) > max_turns * 2:  # 2 messages per turn
            history = history[-max_turns * 2:]

        return self.set(history_key, history, ttl=7200)

    def get_history(self, session_id: str, limit: int = 25) -> List[Dict]:
        """Get conversation history."""
        history = self.get(f"history:{session_id}", [])
        return history[-limit * 2:] if limit else history

    def clear_history(self, session_id: str) -> bool:
        """Clear conversation history."""
        return self.delete(f"history:{session_id}")

    # === Agent State Caching ===

    def cache_agent_state(
        self,
        session_id: str,
        agent_name: str,
        state: Dict[str, Any],
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """Cache agent intermediate state."""
        return self.set(f"agent:{session_id}:{agent_name}", state, ttl)

    def get_agent_state(self, session_id: str, agent_name: str) -> Optional[Dict]:
        """Get cached agent state."""
        return self.get(f"agent:{session_id}:{agent_name}")

    # === Query Caching ===

    def cache_query_result(
        self,
        query_hash: str,
        result: Any,
        ttl: int = 300  # 5 minutes
    ) -> bool:
        """Cache a query result."""
        return self.set(f"query:{query_hash}", result, ttl)

    def get_cached_query(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        return self.get(f"query:{query_hash}")

    # === Rate Limiting ===

    def check_rate_limit(
        self,
        user_id: str,
        action: str,
        limit: int = 10,
        window: int = 60
    ) -> bool:
        """Check if user is within rate limit."""
        key = f"rate:{user_id}:{action}"

        if self.client:
            try:
                current = self.client.get(self._key(key))
                if current is None:
                    self.client.setex(self._key(key), window, 1)
                    return True
                elif int(current) < limit:
                    self.client.incr(self._key(key))
                    return True
                return False
            except:
                pass

        return True  # Allow if Redis not available


# Singleton instance
_stm: Optional[RedisSTM] = None


def get_stm() -> RedisSTM:
    """Get or create STM singleton."""
    global _stm
    if _stm is None:
        _stm = RedisSTM()
    return _stm
