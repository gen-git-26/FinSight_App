from __future__ import annotations
import json
import os
import streamlit as st

from agent import build_agent
from mcp.manager import MCPServer
from tools.mcp_router import mcp_auto

st.set_page_config(page_title="FinSight — Fusion RAG + MCP", layout="wide")

st.title("⚡ FinSight — Fusion RAG + MCP")

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

prompt = st.chat_input("Ask about stocks or crypto… e.g., 'Compare TSLA vs AAPL margins' or 'What is BTC market cap?' ")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent()

    # Lightweight auto‑prefetch: kick mcp_auto once before answering to seed RAG
    try:
        with st.spinner("Fetching data via MCP (if relevant)…"):
            _ = mcp_auto(query=prompt)
    except Exception as e:
        st.warning(f"MCP prefetch skip: {e}")

    with st.spinner("Thinking…"):
        answer = agent.run(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)