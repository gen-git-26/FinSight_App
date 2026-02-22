# tests/test_qdrant_rag.py
"""Tests for Qdrant RAG integration."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rag.qdrant_client import HybridQdrant


def test_qdrant_health_check_method_exists():
    """HybridQdrant should have health_check method."""
    qdr = HybridQdrant()
    assert hasattr(qdr, 'health_check')
    assert callable(qdr.health_check)


def test_qdrant_health_check_returns_dict():
    """health_check should return dict with expected keys."""
    qdr = HybridQdrant()
    result = qdr.health_check()
    assert isinstance(result, dict)
    assert "connected" in result
    assert "collection_exists" in result
    assert "collection_name" in result
    assert "point_count" in result
    assert "error" in result
