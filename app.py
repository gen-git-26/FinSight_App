# app.py
from __future__ import annotations
import os, json
import streamlit as st
import dotenv
dotenv.load_dotenv()

from tools.answer import answer_core
from mcp_connection.manager import MCPServer
from mcp_connection.startup import startup_mcp_servers, get_manager
from tools.mcp_router import route_and_call

# -----------------------------
# Theme & assets
# -----------------------------
LOGO_PATH = "/workspaces/new_test/data/logo.png"   # FinSight logo
BOT_ICON_PATH = "/workspaces/new_test/data/bot_icon.png"                            # Agent icon

PRIMARY_MINT = "#9AF8CC"
TEXT_MAIN   = "#FFFFFF"   
BG_DARK     = "#0D0F10"
CARD_DARK   = "#121416"
BORDER      = "#2A2F33"

st.set_page_config(
    page_title="FinSight",
    page_icon=BOT_ICON_PATH,   # אייקון הטאב
    layout="wide"
)

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {BG_DARK};
        color: {TEXT_MAIN};
      }}
      header, .block-container {{ padding-top: 0.6rem; }}

      /* כרטיסים וצ'אט */
      .stExpander, .stChatMessage, .stTextInput, .stTextArea {{
        background: {CARD_DARK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        color: {TEXT_MAIN} !important;
      }}

      /* טקסט בתוך הודעות וצ'אט */
      .stChatMessage p, .stChatMessage span, .stMarkdown, .stMarkdown p, .stMarkdown span {{
        color: {TEXT_MAIN} !important;
      }}

      /* קווי מפריד */
      hr, .stDivider {{ border-color: {BORDER} !important; }}

      /* כפתורים */
      .stButton>button {{
        background: transparent;
        border: 1px solid {PRIMARY_MINT};
        color: {TEXT_MAIN};
        padding: 0.7rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        transition: 120ms ease-in-out;
      }}
      .stButton>button:hover {{ 
        background: rgba(154,248,204,0.08);
      }}

      /* קלטים */
      .stTextInput input, .stTextArea textarea {{
        color: {TEXT_MAIN} !important;
      }}
      .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #CFD8DC !important;
      }}

      /* אזור מותג */
      .brand-sub {{
        color: #BFEFE0; 
        margin-bottom: 0.6rem; 
        font-size: 1.35rem; 
        font-weight: 700;
        letter-spacing: .2px;
      }}

      /* אייקון הבוט ליד הכפתור */
      .agent-icon {{
        width: 40px; height: 40px; border-radius: 12px; 
        border: 1px solid {PRIMARY_MINT}; background: rgba(154,248,204,0.05);
        display: flex; align-items: center; justify-content: center; overflow: hidden;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Branded header
# -----------------------------
col_logo, col_title, col_spacer = st.columns([0.14, 0.86, 0.10], vertical_alignment="center")
with col_logo:
    st.image(LOGO_PATH, use_container_width=True)
with col_title:
    # רק "Smart Financial Agent" ובפונט מוגדל
    st.markdown('<div class="brand-sub">Smart Financial Agent</div>', unsafe_allow_html=True)

# -----------------------------
# MCP controls (sidebar)
# -----------------------------
with st.sidebar:
    st.image(LOGO_PATH, use_container_width=True)
    st.subheader("MCP Server Status")
    servers = MCPServer.from_env()
    manager = get_manager()
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

# -----------------------------
# Session state
# -----------------------------
if "mcp_started" not in st.session_state:
    st.session_state["mcp_started"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Start MCP servers once
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

# -----------------------------
# Agent button with visible icon
# -----------------------------
agent_col1, agent_col2 = st.columns([0.06, 0.94], vertical_alignment="center")
with agent_col1:
    # בלי file:// בתוך HTML. מציגים ישירות עם st.image כדי שלא יחסם בדפדפן.
    with st.container(border=False):
        st.markdown('<div class="agent-icon">', unsafe_allow_html=True)
        st.image(BOT_ICON_PATH, width=28)
        st.markdown('</div>', unsafe_allow_html=True)

with agent_col2:
    run_agent = st.button("Run FinSight Agent", use_container_width=True)

if run_agent:
    st.session_state["messages"].append({"role": "user", "content": "Analyze NVDA earnings and summarize key drivers."})

# -----------------------------
# Chat history (with custom assistant avatar)
# -----------------------------
for m in st.session_state["messages"]:
    if m["role"] == "assistant":
        with st.chat_message("assistant", avatar=BOT_ICON_PATH):
            st.markdown(m["content"])
    elif m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# -----------------------------
# Chat input
# -----------------------------
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
            raw = mcp_payload
            st.subheader("Raw")
            st.code(raw if isinstance(raw, str) else str(raw), language="text")
            try:
                if isinstance(raw, str):
                    start = min([i for i in [raw.find("["), raw.find("{")] if i != -1] or [None])
                    if start is not None:
                        parsed = json.loads(raw[start:])
                        st.subheader("Parsed (best effort)")
                        _show_parsed(parsed)
            except Exception:
                pass

    # Final assistant answer (shows with bot avatar)
    with st.chat_message("assistant", avatar=BOT_ICON_PATH):
        try:
            out = answer_core(prompt)
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
