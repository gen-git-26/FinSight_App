# infrastructure/validity.py
"""
ValidityClass — maps data types to freshness windows.

Windows are anchored to financial cycles, not arbitrary TTLs.
All times are in seconds. None means permanent (user_preference).
"""
from __future__ import annotations
from enum import Enum
from typing import Optional


class ValidityClass(str, Enum):
    """Data type → validity window mapping."""
    PRICE_SNAPSHOT     = "price_snapshot"      # 1 hour  (intra-day equity)
    END_OF_DAY_PRICE   = "end_of_day_price"    # 48 hours (yesterday close)
    BREAKING_NEWS      = "breaking_news"       # 72 hours (market-moving events)
    NEWS_SENTIMENT     = "news_sentiment"      # 7 days   (aggregated sentiment)
    SESSION_MEMORY     = "session_memory"      # 3 days   (raw conversation turns)
    SESSION_SUMMARY    = "session_summary"     # 30 days  (summarised insights)
    TRADING_DECISION   = "trading_decision"    # 30 days  (swing default; see horizon)
    FUNDAMENTAL_DATA   = "fundamental_data"    # 90 days  (quarterly cycle)
    BEHAVIORAL_PATTERN = "behavioral_pattern"  # 180 days (slowly evolving habits)
    USER_PREFERENCE    = "user_preference"     # permanent (until explicitly changed)

    @property
    def window_seconds(self) -> Optional[int]:
        """Return validity window in seconds, or None for permanent."""
        windows = {
            ValidityClass.PRICE_SNAPSHOT:     1 * 3600,
            ValidityClass.END_OF_DAY_PRICE:   48 * 3600,
            ValidityClass.BREAKING_NEWS:      72 * 3600,
            ValidityClass.NEWS_SENTIMENT:     7 * 86400,
            ValidityClass.SESSION_MEMORY:     3 * 86400,
            ValidityClass.SESSION_SUMMARY:    30 * 86400,
            ValidityClass.TRADING_DECISION:   30 * 86400,
            ValidityClass.FUNDAMENTAL_DATA:   90 * 86400,
            ValidityClass.BEHAVIORAL_PATTERN: 180 * 86400,
            ValidityClass.USER_PREFERENCE:    None,
        }
        return windows[self]


# Horizon → window mapping for trading decisions
_HORIZON_WINDOWS = {
    "day":       7 * 86400,
    "swing":     30 * 86400,
    "long_term": 180 * 86400,
}


def horizon_window_seconds(horizon: str) -> int:
    """Return validity window for a trading horizon. Defaults to swing (30d)."""
    return _HORIZON_WINDOWS.get(horizon, 30 * 86400)


def compute_valid_until(
    validity_class: ValidityClass,
    as_of_epoch: int,
    horizon: Optional[str] = None
) -> Optional[int]:
    """
    Compute valid_until as epoch integer.

    Args:
        validity_class: The ValidityClass for this data type.
        as_of_epoch: When the fact was true (epoch seconds).
        horizon: For TRADING_DECISION only — 'day', 'swing', 'long_term'.

    Returns:
        Epoch int (valid_until), or None if permanent.
    """
    if validity_class == ValidityClass.USER_PREFERENCE:
        return None

    if validity_class == ValidityClass.TRADING_DECISION and horizon:
        return as_of_epoch + horizon_window_seconds(horizon)

    window = validity_class.window_seconds
    if window is None:
        return None
    return as_of_epoch + window
