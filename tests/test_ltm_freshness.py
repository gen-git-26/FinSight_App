# tests/test_ltm_freshness.py
"""Tests for LTM freshness and validity enforcement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timezone


# === Task 2: ValidityClass enum ===

def test_validity_class_window_seconds_price_snapshot():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.PRICE_SNAPSHOT.window_seconds == 3600

def test_validity_class_window_seconds_end_of_day():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.END_OF_DAY_PRICE.window_seconds == 48 * 3600

def test_validity_class_window_seconds_fundamental():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.FUNDAMENTAL_DATA.window_seconds == 90 * 86400

def test_validity_class_user_preference_is_permanent():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.USER_PREFERENCE.window_seconds is None

def test_compute_valid_until_returns_epoch_int_for_finite_class():
    from infrastructure.validity import ValidityClass, compute_valid_until
    as_of = int(datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    result = compute_valid_until(ValidityClass.PRICE_SNAPSHOT, as_of)
    assert result == as_of + 3600

def test_compute_valid_until_returns_none_for_permanent():
    from infrastructure.validity import ValidityClass, compute_valid_until
    as_of = int(datetime(2026, 3, 2, tzinfo=timezone.utc).timestamp())
    assert compute_valid_until(ValidityClass.USER_PREFERENCE, as_of) is None

def test_validity_class_trading_decision_default_is_swing():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.TRADING_DECISION.window_seconds == 30 * 86400

def test_horizon_window_seconds_day():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("day") == 7 * 86400

def test_horizon_window_seconds_long_term():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("long_term") == 180 * 86400

def test_horizon_window_seconds_unknown_defaults_to_swing():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("unknown_value") == 30 * 86400


# === Task 3: INIT_SQL schema migration ===

def test_init_sql_contains_validity_columns():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "validity_class" in INIT_SQL
    assert "valid_for_context_until" in INIT_SQL
    assert "as_of" in INIT_SQL
    assert "source" in INIT_SQL

def test_init_sql_contains_trigger():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "set_validity_window" in INIT_SQL
    assert "BEFORE INSERT" in INIT_SQL

def test_init_sql_contains_check_constraint():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "chk_valid_after_as_of" in INIT_SQL

def test_init_sql_contains_compound_index():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "idx_td_user_class_valid" in INIT_SQL
    assert "idx_cm_user_session_valid" in INIT_SQL

def test_init_sql_contains_unique_constraint_user_patterns():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "UNIQUE (user_id, pattern_type)" in INIT_SQL


# === Task 4: Summary tables ===

def test_summary_sql_contains_validity_columns():
    from infrastructure.postgres_summaries import SUMMARY_TABLES_SQL
    assert "validity_class" in SUMMARY_TABLES_SQL
    assert "valid_for_context_until" in SUMMARY_TABLES_SQL
    assert "as_of" in SUMMARY_TABLES_SQL
