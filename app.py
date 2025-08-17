# new_test/app.py
from __future__ import annotations
import json
import os
import streamlit as st

from agent.agent import build_agent
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto_fn   

st.set_page_config(page_title="FinSight", layout="wide")
st.title("⚡ FinSight")

def _pretty_answer(res) -> str:
    """Extracts a human-readable answer from various response formats."""
    # 1) simple dict with "answer" key
    if isinstance(res, dict) and "answer" in res and isinstance(res["answer"], str):
        return res["answer"]
    # 2) check if res has a content attribute
    content = getattr(res, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # 3) try to parse res as JSON
    if isinstance(res, str):
        try:
            j = json.loads(res)
            if isinstance(j, dict) and "answer" in j:
                return j["answer"]
        except Exception:
            return res
    # 4) check if res has messages attribute
    msgs = getattr(res, "messages", None)
    if isinstance(msgs, list):
        for m in reversed(msgs):
            mc = getattr(m, "content", None)
            if isinstance(mc, str) and '"answer"' in mc:
                try:
                    j = json.loads(mc)
                    if "answer" in j:
                        return j["answer"]
                except Exception:
                    pass
    # 5) fallback
    return str(res)

# Sidebar
servers = MCPServer.from_env()
with st.sidebar:
    st.header("MCP Servers")
    if not servers:
        st.info("No MCP servers configured. Set MCP_SERVERS in .env")
    for name, srv in servers.items():
        st.write(f"**{name}** → `{' '.join(srv.command)}`")
        if srv.env:
            with st.expander("Env"):
                for k, v in srv.env.items():
                    st.code(f"{k}={v}")

st.subheader("Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about stocks or crypto…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent()

    # Prefetch MCP data
    try:
        with st.spinner("Fetching data via MCP…"):
            _ = mcp_auto_fn(prompt)
    except Exception as e:
        st.warning(f"MCP prefetch skip: {e}")

    with st.spinner("Thinking…"):
        res = agent.run(prompt)

    text = _pretty_answer(res)
    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)
