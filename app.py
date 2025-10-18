# app.py
from __future__ import annotations

import os
import json
import streamlit as st
import dotenv

dotenv.load_dotenv()

from tools.answer import answer_core
from mcp_connection.manager import MCPServer
from mcp_connection.startup import startup_mcp_servers, get_manager
from tools.mcp_router import route_and_call

# -----------------------------
# Streamlit setup
# -----------------------------

st.set_page_config(page_title="FinSight", layout="wide")
st.title("⚡ FinSight")

# Session state init
if "mcp_started" not in st.session_state:
    st.session_state["mcp_started"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Start MCP servers
if not st.session_state["mcp_started"]:
    with st.spinner("Starting MCP servers..."):
        results = startup_mcp_servers()
        st.session_state["mcp_started"] = True
        st.session_state["startup_results"] = results
        if results:
            ok, total = sum(results.values()), len(results)
            if ok == total:
                st.success(f"All {ok}/{total} MCP servers started successfully!")
            else:
                st.warning(f"{ok}/{total} MCP servers started")
        else:
            st.info("ℹ No MCP servers configured")

servers = MCPServer.from_env()
manager = get_manager()

# Sidebar server status
with st.sidebar:
    st.header("MCP Server Status")
    if not servers:
        st.info("No MCP servers configured. Set MCP_SERVERS in .env")
    else:
        status = manager.get_server_status()
        for name, _srv in servers.items():
            st.write(f"**{name}** {status.get(name, 'Unknown')}")
        st.divider()
        if st.button("Restart All"):
            with st.spinner("Restarting servers..."):
                manager.stop_all_servers()
                st.session_state["startup_results"] = startup_mcp_servers()
            st.rerun()

# Render chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about stocks, crypto, options, or fundamentals...")

if prompt:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call MCP router directly for a raw view
    try:
        mcp_payload = route_and_call(prompt)
    except Exception as e:
        mcp_payload = {"error": f"Route error: {e}"}

    # Live MCP inspector
    with st.expander("🔌 Live MCP"):
        def _show_parsed(obj):
            try:
                import pandas as pd
            except Exception:
                pd = None
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and pd is not None:
                st.dataframe(pd.DataFrame(obj))
            elif isinstance(obj, (dict, list)):
                st.json(obj)
            else:
                st.code(str(obj), language="text")

        if isinstance(mcp_payload, dict):
            if mcp_payload.get("error"):
                st.error(mcp_payload["error"])
            else:
                st.subheader("Route")
                st.json(mcp_payload.get("route", {}))
                st.subheader("Parsed")
                if mcp_payload.get("parsed") is not None:
                    _show_parsed(mcp_payload.get("parsed"))
                else:
                    st.info("No JSON payload. See Raw below.")
                st.subheader("Raw")
                st.code(mcp_payload.get("raw", ""), language="json")
        else:
            # Backward compatibility: if router returns a string
            raw = mcp_payload
            st.subheader("Raw")
            st.code(raw if isinstance(raw, str) else str(raw), language="text")
            # Try to parse JSON from the raw string (best effort)
            try:
                # Heuristic: extract substring from first '[' or '{'
                if isinstance(raw, str):
                    start = min([i for i in [raw.find("[") , raw.find("{")] if i != -1] or [None])
                    if start is not None:
                        parsed = json.loads(raw[start:])
                        st.subheader("Parsed (best effort)")
                        _show_parsed(parsed)
            except Exception:
                pass

    # Now compute the final assistant answer
    with st.chat_message("assistant"):
        try:
            out = answer_core(prompt)  # answer_core internally decides MCP vs RAG
            txt = (out or {}).get("answer", "")
            st.markdown(txt if txt else "*No answer text returned*")
            st.session_state["messages"].append({"role": "assistant", "content": txt})

            snips = (out or {}).get("snippets") or []
            if snips:
                with st.expander(f"Sources Used ({len(snips)} snippets)"):
                    for i, s in enumerate(snips[:10], 1):
                        sym = s.get("symbol") or "N/A"
                        dt_ = s.get("date") or "N/A"
                        src = s.get("source") or "unknown"
                        prev = (s.get("text") or "")[:200]
                        st.write(f"**{i}.** [{src}] **{sym}** ({dt_})")
                        st.write(f"_{prev}..._")
                        st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})
