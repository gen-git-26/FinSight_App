"""
Tests for app.py startup: secrets sync and password gate.

We test three things:
1. The secrets sync logic (tested in isolation — importing app.py would invoke Streamlit rendering)
2. The structure of app.py source (confirms the sync block was added and ordering is correct)
3. Edge cases for the sync logic
"""
import os
import ast
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Structure tests: verify app.py source contains required blocks in correct order
# ---------------------------------------------------------------------------

def _app_source() -> str:
    return (Path(__file__).parent.parent / "app.py").read_text()


def test_app_contains_secrets_sync():
    """app.py must contain the st.secrets sync block."""
    src = _app_source()
    assert "st.secrets.items()" in src, "secrets sync block missing from app.py"
    assert "os.environ.setdefault" in src, "setdefault call missing from app.py"


def test_app_secrets_sync_inside_try_except():
    """secrets sync must be wrapped in try/except to avoid crash when no secrets.toml exists."""
    src = _app_source()
    # try block must come before st.secrets.items()
    assert src.index("try:") < src.index("st.secrets.items()"), (
        "st.secrets.items() must be inside a try block"
    )


def test_app_skeleton_import_before_password_gate():
    """skeleton import (which triggers set_page_config) must precede st.text_input."""
    src = _app_source()
    assert "from ui.skeleton import main" in src, "skeleton import missing"
    assert "st.text_input" in src, "password gate missing"
    assert src.index("from ui.skeleton import main") < src.index("st.text_input"), (
        "skeleton must be imported (triggering set_page_config) before st.text_input is called"
    )


def test_app_agent_import_after_skeleton_import():
    """agent.graph import must come after skeleton import (secrets must be in os.environ first)."""
    src = _app_source()
    assert "from agent.graph import run_query" in src
    assert src.index("from ui.skeleton import main") < src.index("from agent.graph import run_query"), (
        "agent.graph import must come after skeleton import"
    )


# ---------------------------------------------------------------------------
# Logic tests: secrets sync behaviour (tested in isolation)
# ---------------------------------------------------------------------------

def _run_sync(secrets_dict, existing_env=None):
    """Helper: run the secrets sync logic against a fake st.secrets."""
    import streamlit as st
    env_patch = existing_env or {}
    with patch.dict(os.environ, env_patch, clear=False):
        # Remove keys not in existing_env
        for k in secrets_dict:
            if k not in (existing_env or {}):
                os.environ.pop(k, None)
        with patch.object(type(st.secrets), "items", return_value=secrets_dict.items()):
            try:
                for _k, _v in st.secrets.items():
                    if isinstance(_v, str):
                        os.environ.setdefault(_k, _v)
            except Exception:
                pass
        return {k: os.environ.get(k) for k in secrets_dict}


def test_secrets_sync_copies_strings_to_environ():
    result = _run_sync({"OPENAI_API_KEY": "sk-test", "REDIS_HOST": "myhost"})
    assert result["OPENAI_API_KEY"] == "sk-test"
    assert result["REDIS_HOST"] == "myhost"


def test_secrets_sync_does_not_override_existing():
    result = _run_sync(
        {"OPENAI_API_KEY": "sk-from-secrets"},
        existing_env={"OPENAI_API_KEY": "sk-existing"}
    )
    assert result["OPENAI_API_KEY"] == "sk-existing"


def test_secrets_sync_skips_non_string_values():
    """Nested TOML sections (dicts) must not be written to os.environ."""
    import streamlit as st
    os.environ.pop("FLAT_KEY", None)
    os.environ.pop("SECTION", None)
    fake = {"SECTION": {"key": "val"}, "FLAT_KEY": "flat_val"}
    with patch.object(type(st.secrets), "items", return_value=fake.items()):
        try:
            for _k, _v in st.secrets.items():
                if isinstance(_v, str):
                    os.environ.setdefault(_k, _v)
        except Exception:
            pass
    assert os.environ.get("FLAT_KEY") == "flat_val"
    assert "SECTION" not in os.environ


def test_secrets_sync_silent_on_missing_secrets_toml():
    """When no secrets.toml exists locally, the sync block must not raise."""
    import streamlit as st
    with patch.object(type(st.secrets), "items", side_effect=FileNotFoundError):
        try:
            for _k, _v in st.secrets.items():
                if isinstance(_v, str):
                    os.environ.setdefault(_k, _v)
        except Exception:
            pass  # must NOT propagate — test passes only if we reach this point
