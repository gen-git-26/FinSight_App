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
LOGO_PATH = "/mnt/data/logo.png"  # FinSight logo
BOT_ICON_PATH = "/mnt/data/bot_icon.png"                           # Agent icon

PRIMARY_MINT = "#9AF8CC"   
TEXT_MAIN   = "#E6FFF4"
BG_DARK     = "#0D0F10"
CARD_DARK   = "#121416"
BORDER      = "#2A2F33"

st.set_page_config(
    page_title="FinSight",
    page_icon=BOT_ICON_PATH,  # אייקון לשונית
    layout="wide"
)

# Global CSS (שחור מלא + סטייל כפתורים ותיבות)
st.markdown(
    f"""
    <style>
      /* Background + text */
      .stApp {{
        background: {BG_DARK};
        color: {TEXT_MAIN};
      }}
      /* Hide default Streamlit header gap */
      header, .block-container {{
        padding-top: 0.6rem;
      }}

      /* Card-like containers (expanders/chat) */
      .stExpander, .stChatMessage, .stTextInput, .stTextArea {{
        background: {CARD_DARK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
      }}

      /* Nice separators */
      .st-emotion-cache-1cvow4s hr, .stDivider {{
        border-color: {BORDER} !important;
      }}

      /* Buttons */
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
        border-color: {PRIMARY_MINT};
      }}
      .stButton>button:focus {{ outline: none; }}

      /* Chat input placeholder color */
      .stTextInput input::placeholder {{
        color: #7aa996;
      }}

      /* Title row spacing */
      .finsight-title {{
        display: flex; align-items: center; gap: 0.75rem;
        margin-bottom: 0.25rem;
      }}
      .brand-sub {{
        color: #9bcab8; margin-bottom: 0.75rem; font-size: 0.95rem;
      }}

      /* Agent row */
      .agent-row {{
        display: flex; align-items: center; gap: 0.6rem; margin: 0.4rem 0 1rem 0;
      }}
      .agent-icon {{
        width: 40px; height: 40px; border-radius: 12px; 
        border: 1px solid {PRIMARY_MINT}; background: rgba(154,248,204,0.05);
        display: flex; align-items: center; justify-content: center; overflow: hidden;
      }}
      .agent-icon img {{ width: 70%; height: 70%; object-fit: contain; filter: brightness(0.9); }}
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
    st.markdown('<div class="brand-sub">Smart Financial Agent • Fusion RAG • Live Market Tools</div>', unsafe_allow_html=True)

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
# Agent button (with icon)
# -----------------------------
agent_col1, agent_col2 = st.columns([0.06, 0.94], vertical_alignment="center")
with agent_col1:
    st.markdown(
        f"""
        <div class="agent-row">
          <div class="agent-icon">
            <img src="file://{BOT_ICON_PATH}" alt="bot icon"/>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with agent_col2:
    run_agent = st.button("Run FinSight Agent", use_container_width=True)

# If agent button clicked, seed an example action (optional)
if run_agent:
    st.session_state["messages"].append({"role": "user", "content": "Analyze NVDA earnings and summarize key drivers."})

# -----------------------------
# Chat history
# -----------------------------
for m in st.session_state["messages"]:
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

    # Final assistant answer
    with st.chat_message("assistant"):
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
