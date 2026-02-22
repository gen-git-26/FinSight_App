# app.py
"""
FinSight - Multi-Agent Financial Analysis System

Powered by LangGraph with A2A (Agent-to-Agent) architecture:
- Standard Flow: Router → Fetcher/Crypto → Analyst → Composer
- Trading Flow: Router → Fetcher → Analysts Team → Researchers → Trader → Risk Manager → Fund Manager → Composer
"""
from __future__ import annotations
import os, json
import streamlit as st
import dotenv
dotenv.load_dotenv()

# New multi-agent system
from agent.graph import run_query, stream_query, get_graph
from agent.nodes.router import classify_trading_subtype, TRADING_KEYWORDS

# -----------------------------
# Theme & assets
# -----------------------------
LOGO_PATH = "ui/logo.png"
BOT_ICON_PATH = "ui/bot_icon.png"

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
# Sidebar - Agent Info
# -----------------------------
with st.sidebar:
    st.image(LOGO_PATH, use_container_width=True)
    st.subheader("Multi-Agent System")

    st.markdown("""
    **Standard Flow:**
    - Router → Fetcher/Crypto → Analyst → Composer

    **Granular Trading Flow:**
    - "What's the P/E?" → Fundamental Analyst only
    - "Is it oversold?" → Technical Analyst only
    - "Any news?" → News Analyst only
    - "What's the sentiment?" → Sentiment Analyst only

    **Full Trading Flow (A2A):**
    - "Should I buy AAPL?" triggers all 6 agents:
    - 4 Analysts → Bull vs Bear → Trader → Risk → Fund Manager
    """)

    st.divider()
    st.caption("Powered by LangGraph")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "default"

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

    # Process with multi-agent system
    with st.chat_message("assistant", avatar=BOT_ICON_PATH):
        try:
            # Detect query type for appropriate progress indicator
            prompt_lower = prompt.lower()
            is_trading = any(kw in prompt_lower for kw in TRADING_KEYWORDS)

            if is_trading:
                # Determine if granular or full trading flow
                trading_subtype = classify_trading_subtype(prompt)
                is_granular = trading_subtype != "full_trading"

                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    if is_granular:
                        # Granular flow: single analyst
                        analyst_name = trading_subtype.title()
                        st.info(f"Running {analyst_name} Analysis...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        stages = [
                            (20, "Routing query..."),
                            (50, "Fetching market data..."),
                            (80, f"Running {analyst_name} Analyst..."),
                            (100, "Composing response...")
                        ]
                    else:
                        # Full A2A trading flow
                        st.info("Processing with Full TradingAgents A2A flow...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        stages = [
                            (10, "Routing query..."),
                            (20, "Fetching market data..."),
                            (40, "Running 4 analyst reports..."),
                            (60, "Bull vs Bear research debate..."),
                            (70, "Trader making decision..."),
                            (85, "Risk management assessment..."),
                            (95, "Fund manager approval..."),
                            (100, "Composing response...")
                        ]

                    # Simulate initial progress
                    import time
                    for pct, msg in stages[:2]:
                        progress_bar.progress(pct)
                        status_text.text(msg)
                        time.sleep(0.1)

                # Run the actual query
                response = run_query(prompt, user_id=st.session_state["user_id"])

                # Clear progress
                progress_placeholder.empty()
            else:
                # Standard flow with simple spinner
                with st.spinner("Analyzing..."):
                    response = run_query(prompt, user_id=st.session_state["user_id"])

            # Display response
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error: {e}")
            print(f"[app] Assistant response error: {e}")
            import traceback
            traceback.print_exc()
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})