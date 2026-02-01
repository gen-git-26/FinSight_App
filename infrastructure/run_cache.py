# infrastructure/run_cache.py
"""
Run Cache - Caches tool/API results within a run to avoid duplicate calls.

In A2A architecture, multiple analysts may request the same data (OHLCV, quotes, news).
Run Cache deduplicates these calls dramatically reducing latency and API costs.

Keys format: run:{run_id}:{tool}:{ticker}:{params_hash}
TTL: Short (5-15 minutes) - only valid within a single run
"""
from __future__ import annotations

import os
import json
import hashlib
from typing import Optional, Dict, Any, List, Callable, TypeVar, Awaitable
from dataclasses import dataclass
from functools import wraps

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from infrastructure.memory_types import RunCacheKey


T = TypeVar('T')


@dataclass
class CacheConfig:
    """Run cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 1  # Separate DB from STM
    password: Optional[str] = None
    prefix: str = "finsight:run:"
    default_ttl: int = 300  # 5 minutes

    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_CACHE_DB", "1")),
            password=os.getenv("REDIS_PASSWORD"),
            prefix=os.getenv("REDIS_CACHE_PREFIX", "finsight:run:"),
            default_ttl=int(os.getenv("REDIS_CACHE_TTL", "300")),
        )


class RunCache:
    """
    Tool results cache for A2A deduplication.

    Caches:
    - Stock quotes
    - OHLCV data
    - News fetches
    - Fundamentals
    - TA computations

    Example keys:
    - run:abc123:quote:AAPL
    - run:abc123:ohlcv:TSLA:1d
    - run:abc123:news:NVDA:5
    """

    # TTL by data type (seconds)
    TTL_MAP = {
        "quote": 60,          # 1 minute - prices change fast
        "ohlcv": 300,         # 5 minutes - historical data stable
        "news": 600,          # 10 minutes - news stable
        "fundamentals": 3600, # 1 hour - fundamentals very stable
        "options": 120,       # 2 minutes - options change
        "ta": 300,            # 5 minutes - TA computations
        "crypto": 30,         # 30 seconds - crypto volatile
    }

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig.from_env()
        self._client: Optional[redis.Redis] = None
        self._fallback: Dict[str, Any] = {}

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
                self._client.ping()
            except Exception as e:
                print(f"[RunCache] Redis connection failed: {e}")
                self._client = None

        return self._client

    def _make_key(self, run_id: str, tool: str, ticker: Optional[str] = None,
                  params: Optional[Dict] = None) -> str:
        """Generate cache key."""
        parts = [self.config.prefix, run_id, tool]
        if ticker:
            parts.append(ticker.upper())
        if params:
            # Hash params for consistent keys
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            parts.append(params_hash)
        return ":".join(parts)

    def _get_ttl(self, tool: str) -> int:
        """Get TTL for tool type."""
        for key, ttl in self.TTL_MAP.items():
            if key in tool.lower():
                return ttl
        return self.config.default_ttl

    # === Core Operations ===

    def get(self, run_id: str, tool: str, ticker: Optional[str] = None,
            params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached result."""
        key = self._make_key(run_id, tool, ticker, params)

        if self.client:
            try:
                value = self.client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"[RunCache] Get failed: {e}")

        return self._fallback.get(key)

    def set(self, run_id: str, tool: str, value: Any,
            ticker: Optional[str] = None, params: Optional[Dict] = None,
            ttl: Optional[int] = None) -> bool:
        """Cache a result."""
        key = self._make_key(run_id, tool, ticker, params)
        ttl = ttl or self._get_ttl(tool)

        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            serialized = json.dumps(str(value))

        if self.client:
            try:
                self.client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                print(f"[RunCache] Set failed: {e}")

        self._fallback[key] = value
        return True

    def exists(self, run_id: str, tool: str, ticker: Optional[str] = None,
               params: Optional[Dict] = None) -> bool:
        """Check if key exists."""
        key = self._make_key(run_id, tool, ticker, params)

        if self.client:
            try:
                return self.client.exists(key) > 0
            except:
                pass

        return key in self._fallback

    def invalidate_run(self, run_id: str) -> int:
        """Invalidate all cache entries for a run."""
        pattern = f"{self.config.prefix}{run_id}:*"
        deleted = 0

        if self.client:
            try:
                keys = self.client.keys(pattern)
                if keys:
                    deleted = self.client.delete(*keys)
            except Exception as e:
                print(f"[RunCache] Invalidate failed: {e}")

        # Clean fallback
        to_delete = [k for k in self._fallback if k.startswith(f"{self.config.prefix}{run_id}:")]
        for k in to_delete:
            del self._fallback[k]
            deleted += 1

        return deleted

    # === Convenience Methods ===

    def get_quote(self, run_id: str, ticker: str) -> Optional[Dict]:
        """Get cached quote."""
        return self.get(run_id, "quote", ticker)

    def set_quote(self, run_id: str, ticker: str, quote: Dict) -> bool:
        """Cache a quote."""
        return self.set(run_id, "quote", quote, ticker)

    def get_ohlcv(self, run_id: str, ticker: str, interval: str = "1d") -> Optional[List]:
        """Get cached OHLCV data."""
        return self.get(run_id, "ohlcv", ticker, {"interval": interval})

    def set_ohlcv(self, run_id: str, ticker: str, data: List, interval: str = "1d") -> bool:
        """Cache OHLCV data."""
        return self.set(run_id, "ohlcv", data, ticker, {"interval": interval})

    def get_news(self, run_id: str, ticker: str, limit: int = 10) -> Optional[List]:
        """Get cached news."""
        return self.get(run_id, "news", ticker, {"limit": limit})

    def set_news(self, run_id: str, ticker: str, news: List, limit: int = 10) -> bool:
        """Cache news."""
        return self.set(run_id, "news", news, ticker, {"limit": limit})

    def get_fundamentals(self, run_id: str, ticker: str) -> Optional[Dict]:
        """Get cached fundamentals."""
        return self.get(run_id, "fundamentals", ticker)

    def set_fundamentals(self, run_id: str, ticker: str, data: Dict) -> bool:
        """Cache fundamentals."""
        return self.set(run_id, "fundamentals", data, ticker)

    # === Decorator for Auto-Caching ===

    def cached(self, tool: str, ticker_param: str = "ticker"):
        """
        Decorator to auto-cache async function results.

        Usage:
            @cache.cached("quote", ticker_param="symbol")
            async def get_quote(run_id: str, symbol: str) -> Dict:
                ...
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                run_id = kwargs.get("run_id") or (args[0] if args else None)
                ticker = kwargs.get(ticker_param)

                if run_id and ticker:
                    # Check cache first
                    cached = self.get(run_id, tool, ticker)
                    if cached is not None:
                        return cached

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                if run_id and ticker and result is not None:
                    self.set(run_id, tool, result, ticker)

                return result
            return wrapper
        return decorator


# Singleton
_cache: Optional[RunCache] = None


def get_run_cache() -> RunCache:
    """Get or create RunCache singleton."""
    global _cache
    if _cache is None:
        _cache = RunCache()
    return _cache
