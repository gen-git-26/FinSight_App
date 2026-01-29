# datasources/models.py
"""
Data models for unified data sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class DataSourceType(Enum):
    """Types of data sources."""
    MCP = "mcp"
    API = "api"
    CACHE = "cache"


class DataType(Enum):
    """Types of financial data."""
    PRICE = "price"
    QUOTE = "quote"
    FUNDAMENTALS = "fundamentals"
    OPTIONS = "options"
    HISTORICAL = "historical"
    NEWS = "news"
    CRYPTO = "crypto"
    EARNINGS = "earnings"
    INSIDER = "insider"


@dataclass
class StockQuote:
    """Stock quote data."""
    symbol: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    name: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: str = ""


@dataclass
class Fundamentals:
    """Company fundamentals data."""
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    revenue: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    eps: Optional[float] = None
    beta: Optional[float] = None
    source: str = ""


@dataclass
class OptionsData:
    """Options chain data."""
    symbol: str
    expiration: str
    calls: List[Dict[str, Any]] = field(default_factory=list)
    puts: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""


@dataclass
class HistoricalData:
    """Historical price data."""
    symbol: str
    period: str
    interval: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""


@dataclass
class NewsItem:
    """News article data."""
    title: str
    url: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    published: Optional[datetime] = None
    symbol: Optional[str] = None
    sentiment: Optional[str] = None


@dataclass
class CryptoQuote:
    """Cryptocurrency quote data."""
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change_24h: Optional[float] = None
    change_percent_24h: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    source: str = ""


@dataclass
class DataResult:
    """Unified result from any data source."""
    success: bool
    data: Any = None
    data_type: Optional[DataType] = None
    source: str = ""
    source_type: DataSourceType = DataSourceType.API
    error: Optional[str] = None
    raw: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data if not hasattr(self.data, "__dataclass_fields__") else self.data.__dict__,
            "data_type": self.data_type.value if self.data_type else None,
            "source": self.source,
            "source_type": self.source_type.value,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }
