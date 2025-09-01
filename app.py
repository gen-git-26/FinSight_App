# app.py
from __future__ import annotations
import os, json, streamlit as st, dotenv
dotenv.load_dotenv()

from tools.answer import answer_core          # 👈 ליבה יציבה
from mcp_connection.manager import MCPServer
from mcp_connection.startup import startup_mcp_servers, get_manager
from tools.mcp_router import route_and_call   # להציג RAW

st.set_page_config(page_title="FinSight", layout="wide")
st.title("⚡ FinSight")

if "mcp_started" not in st.session_state:
    st.session_state["mcp_started"] = False

if not st.session_state["mcp_started"]:
    with st.spinner("Starting MCP servers..."):
        results = startup_mcp_servers()
        st.session_state["mcp_started"] = True
        st.session_state["startup_results"] = results
        if results:
            ok, total = sum(results.values()), len(results)
            st.success(f"All {ok}/{total} MCP servers started successfully!" if ok==total else f"{ok}/{total} MCP servers started")
        else:
            st.info("ℹNo MCP servers configured")

servers = MCPServer.from_env()
manager = get_manager()

with st.sidebar:
    st.header("MCP Server Status")
    if not servers:
        st.info("No MCP servers configured. Set MCP_SERVERS in .env")
    else:
        status = manager.get_server_status()
        for name, srv in servers.items():
            st.write(f"**{name}** {status.get(name,'Unknown')}")
        st.divider()
        if st.button("Restart All"):
            with st.spinner("Restarting servers..."):
                manager.stop_all_servers()
                st.session_state["startup_results"] = startup_mcp_servers()
            st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about stocks, crypto, or market data...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # הצגת ה-RAW כמו בטרמינל
    try:
        mcp_raw = route_and_call(prompt)
    except Exception as e:
        mcp_raw = f"Route error: {e}"
    with st.expander("🔌 Live MCP (raw)"):
        st.code(mcp_raw, language="json")

    # תשובה יציבה מהליבה
    with st.chat_message("assistant"):
        try:
            out = answer_core(prompt)              # 👈 קריאה יציבה
            txt = (out or {}).get("answer", "")
            st.markdown(txt if txt else "*No answer text returned*")
            st.session_state["messages"].append({"role": "assistant", "content": txt})

            snips = (out or {}).get("snippets") or []
            if snips:
                with st.expander(f"Sources Used ({len(snips)} snippets)"):
                    for i, s in enumerate(snips[:10], 1):
                        sym = s.get("symbol") or "N/A"
                        date = s.get("date") or "N/A"
                        src = s.get("source") or "unknown"
                        prev = (s.get("text") or "")[:150]
                        st.write(f"**{i}.** [{src}] **{sym}** ({date})")
                        st.write(f"_{prev}..._")
                        st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state["messages"].append({"role": "assistant","content": f"Error: {e}"})
