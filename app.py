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
    page_icon=BOT_ICON_PATH,   
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

      /* cards and chat */
      .stExpander, .stChatMessage, .stTextInput, .stTextArea {{
        background: {CARD_DARK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        color: {TEXT_MAIN} !important;
      }}

      /* text user input */
      .stChatMessage p, .stChatMessage span, .stMarkdown, .stMarkdown p, .stMarkdown span {{
        color: {TEXT_MAIN} !important;
      }}

      /* dividers */
      hr, .stDivider {{ border-color: {BORDER} !important; }}

      /* buttons */
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

      /* input */
      .stTextInput input, .stTextArea textarea {{
        color: {TEXT_MAIN} !important;
      }}
      .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #CFD8DC !important;
      }}

      /* brand subtitle */
      .brand-sub {{
        color: #BFEFE0; 
        margin-bottom: 0.6rem; 
        font-size: 1.35rem; 
        font-weight: 700;
        letter-spacing: .2px;
      }}
    
      /*icon inline fix*/
      .align-center {{
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        }}


      /* delete streamlit watermark */
        [data-testid="stImage"] img {{
        margin: 0 !important; 
        display: block; 
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
    st.markdown("<h1 style='margin-bottom: 0.1rem;'>See Beyond The Numbers</h1>", unsafe_allow_html=True)
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
    st.markdown(
        """
        <div style='display:flex; align-items:center; justify-content:center; height:100%;'>
        """,
        unsafe_allow_html=True,
    )
    st.image(BOT_ICON_PATH, width=60)
    st.markdown("</div>", unsafe_allow_html=True)

with agent_col2:
    run_agent = st.button("Run FinSight Agent", use_container_width=True)

if run_agent:
    st.session_state["messages"].append({"role": "user", "content": "Ask me anything..."})

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
            display_type = (out or {}).get("display_type", "text")
            is_dataframe = (out or {}).get("is_dataframe", False)
            
            # Smart display based on content type
            if is_dataframe and display_type == "table":
                # Parse and display as interactive table
                try:
                    import pandas as pd
                    data = json.loads(txt)
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    df = pd.DataFrame(json.loads(data)) if isinstance(data, str) else pd.DataFrame(data)
                    
                    # Format columns nicely
                    with st.container(border=True):
                        st.markdown("### Results")
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                col: st.column_config.NumberColumn(
                                    format="$%.2f" if "price" in col.lower() else "%.4f"
                                )
                                for col in df.columns
                                if df[col].dtype in ['float64', 'int64']
                            }
                        )
                except Exception as e:
                    print(f"[app] Dataframe render error: {e}")
                    st.markdown(txt)
            
            elif display_type == "dict":
                # Formatted key-value display
                with st.container(border=True):
                    st.markdown("###Data Summary")
                    st.markdown(txt)
            
            elif display_type == "error":
                # Error display
                st.error(txt)
            
            else:
                # Default text display
                st.markdown(txt)
            
            # Add metadata section
            meta = (out or {}).get("meta", {})
            if meta:
                with st.expander("Metadata", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**MCP Success**: {meta.get('mcp_success', 'N/A')}")
                        st.write(f"**Display Type**: {meta.get('display_type', 'N/A')}")
                    with col2:
                        st.write(f"**Data Source**: {meta.get('available_servers', [])}")
                        st.write(f"**Is DataFrame**: {meta.get('is_dataframe', False)}")
            
            # Store message
            st.session_state["messages"].append({"role": "assistant", "content": txt})

            # Sources (if available)
            snips = (out or {}).get("snippets") or []
            if snips:
                with st.expander(f"Sources ({len(snips)} snippets)", expanded=False):
                    for i, s in enumerate(snips[:10], 1):
                        sym = s.get("symbol") or "N/A"
                        dt_ = s.get("date") or "N/A"
                        src = s.get("source") or "unknown"
                        prev = (s.get("text") or "")[:200]
                        st.write(f"**{i}.** `{src}` | **{sym}** ({dt_})")
                        st.write(f"_{prev}..._")
                        st.divider()
        
        except Exception as e:
            st.error(f"Error: {e}")
            print(f"[app] Assistant response error: {e}")
            import traceback
            traceback.print_exc()
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})