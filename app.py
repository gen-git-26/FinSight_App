# app.py — drop-in
from __future__ import annotations
import os
import json
import inspect
import asyncio
import streamlit as st
import dotenv
dotenv.load_dotenv()

from agent.agent import build_agent
from mcp_connection.manager import MCPServer
from tools.mcp_router import route_and_call  # פונקציה רגילה (לא כלי של Agno)

st.set_page_config(page_title="FinSight — Fusion RAG + MCP", layout="wide")
st.title("⚡ FinSight — Fusion RAG + MCP")

# --- Sidebar: show MCP servers from env
servers = MCPServer.from_env()
with st.sidebar:
    st.header("MCP Servers")
    if not servers:
        st.info("No MCP servers configured. Set MCP_SERVERS in .env")
    for name, srv in servers.items():
        cmd_str = " ".join(srv.command) if isinstance(srv.command, (list, tuple)) else str(srv.command)
        st.write(f"**{name}** → `{cmd_str}`")
        if srv.env:
            with st.expander("Env"):
                for k, v in srv.env.items():
                    st.code(f"{k}={v}")

# --- Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Prefetch helper (router only; לא קוראים לכלי Agno ישירות)
def prefetch_via_mcp(query: str) -> None:
    try:
        if inspect.iscoroutinefunction(route_and_call):
            asyncio.run(route_and_call(query))
        else:
            route_and_call(query)
    except Exception as e:
        st.info(f"MCP prefetch skip: {e}")

# --- Input
prompt = st.chat_input("Ask about stocks or crypto…")
if prompt:
    st.session_state["messages"].append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent()

    with st.spinner("Thinking…"):
        res = agent.run(prompt)  

    
    # final_text = res
    
    # if isinstance(res, dict) and "answer" in res:
    #     final_text = res["answer"]
    
    # elif hasattr(res, "content"):
    #     final_text = res.content

    # st.session_state["messages"].append({"role":"assistant","content":final_text})
    # with st.chat_message("assistant"):
    #     st.markdown(final_text)

        text = getattr(res, "content", res)
    with st.chat_message("assistant"):
        st.markdown(text)

    # מקורות: חילוץ ה-snippets שמהם נבנתה התשובה
    import json
    def _extract_snippets(run):
        out = []
        for m in getattr(run, "messages", []):
            if getattr(m, "role", "") == "tool" and getattr(m, "content", None):
                try:
                    obj = json.loads(m.content)
                    if "snippets" in obj:
                        out += obj["snippets"]
                except Exception:
                    pass
        return out

    snips = _extract_snippets(res)
    if snips:
        with st.expander("Sources / snippets used"):
            for s in snips[:8]:
                sym  = s.get("symbol") or ""
                date = s.get("date")   or ""
                src  = s.get("source") or ""
                st.write(f"- **{sym}** {date} · _{src}_ — {s['text'][:200]}…")
