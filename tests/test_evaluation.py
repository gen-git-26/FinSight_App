# tests/test_evaluation.py
"""Tests for evaluation metrics module."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_metrics_dataclass_exists():
    """CallMetrics dataclass should exist with expected fields."""
    from evaluation.metrics import CallMetrics
    metrics = CallMetrics(
        agent_name="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=150.5,
        success=True
    )
    assert metrics.agent_name == "test"
    assert metrics.total_tokens == 150


def test_track_metrics_decorator_exists():
    """track_metrics decorator should exist and be callable."""
    from evaluation.metrics import track_metrics
    assert callable(track_metrics)


def test_get_session_metrics_exists():
    """Session metrics functions should exist."""
    from evaluation.metrics import get_session_metrics, clear_session_metrics
    clear_session_metrics()
    metrics = get_session_metrics()
    assert isinstance(metrics, list)


def test_track_metrics_decorator_captures_latency():
    """Decorator should capture latency for decorated functions."""
    from evaluation.metrics import track_metrics, get_session_metrics, clear_session_metrics
    import time

    clear_session_metrics()

    @track_metrics("test_agent")
    def slow_function():
        time.sleep(0.01)
        return "result"

    result = slow_function()
    assert result == "result"

    metrics = get_session_metrics()
    assert len(metrics) == 1
    assert metrics[0].agent_name == "test_agent"
    assert metrics[0].latency_ms >= 10  # At least 10ms
    assert metrics[0].success is True
