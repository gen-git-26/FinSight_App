# app.py — Enhanced with MCP auto-startup
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
from mcp_connection.startup import startup_mcp_servers, get_manager
from tools.mcp_router import route_and_call

st.set_page_config(page_title="FinSight", layout="wide")
st.title("⚡ FinSight")

# Initialize MCP servers on first run
if "mcp_started" not in st.session_state:
    st.session_state["mcp_started"] = False

if not st.session_state["mcp_started"]:
    with st.spinner("Starting MCP servers..."):
        startup_results = startup_mcp_servers()
        st.session_state["mcp_started"] = True
        st.session_state["startup_results"] = startup_results
        
        # Show startup results
        if startup_results:
            success_count = sum(startup_results.values())
            total_count = len(startup_results)
            if success_count == total_count:
                st.success(f"All {total_count} MCP servers started successfully!")
            else:
                st.warning(f"{success_count}/{total_count} MCP servers started successfully")
        else:
            st.info("ℹNo MCP servers configured")

# --- Enhanced Sidebar with server status ---
servers = MCPServer.from_env()
manager = get_manager()

with st.sidebar:
    st.header("MCP Server Status")
    
    if not servers:
        st.info("No MCP servers configured. Set MCP_SERVERS in .env")
    else:
        server_status = manager.get_server_status()
        
        for name, srv in servers.items():
            status = server_status.get(name, "Unknown")
            st.write(f"**{name}** {status}")
            
            # Show command and env in expander
            with st.expander(f"Config: {name}"):
                cmd_str = " ".join(srv.command) if isinstance(srv.command, (list, tuple)) else str(srv.command)
                st.code(f"Command: {cmd_str}")
                
                if srv.env:
                    st.write("**Environment:**")
                    for k, v in srv.env.items():
                        # Hide sensitive values
                        display_v = v if len(v) < 20 else f"{v[:10]}...{v[-4:]}"
                        st.code(f"{k}={display_v}")
        
        # Server management buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Restart All"):
                with st.spinner("Restarting servers..."):
                    manager.stop_all_servers()
                    startup_results = startup_mcp_servers()
                    st.session_state["startup_results"] = startup_results
                st.rerun()
        
        with col2:
            if st.button("Status"):
                st.rerun()

    # Show capabilities
    st.header("Agent Capabilities")
    capabilities = []
    if "yfinance" in servers:
        capabilities.append("Yahoo Finance: Live stock data")
    if "financial-datasets" in servers:
        capabilities.append("Financial Datasets: Stocks & crypto")
    if "coinmarketcap" in servers:
        capabilities.append("CoinMarketCap: Crypto prices")
    
    if capabilities:
        for cap in capabilities:
            st.write(f"• {cap}")
    else:
        st.write("• Static Finnhub & Alpha Vantage data")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about stocks, crypto, or market data...")

if prompt:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and fetching data..."):
            try:
                agent = build_agent()
                res = agent.run(prompt)
                
                # Extract response text
                if hasattr(res, 'content'):
                    response_text = res.content
                elif isinstance(res, str):
                    response_text = res
                else:
                    response_text = str(res)
                
                st.markdown(response_text)
                
                # Add to chat history
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": response_text
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": error_msg
                })

    # Show sources if available
    def _extract_snippets(run):
        """Extract snippets from agent run for source transparency."""
        out = []
        if hasattr(run, "messages"):
            for m in getattr(run, "messages", []):
                if getattr(m, "role", "") == "tool" and getattr(m, "content", None):
                    try:
                        obj = json.loads(m.content)
                        if "snippets" in obj:
                            out += obj["snippets"]
                    except Exception:
                        pass
        return out

    if 'res' in locals():
        snips = _extract_snippets(res)
        if snips:
            with st.expander(f"Sources Used ({len(snips)} snippets)"):
                for i, s in enumerate(snips[:10], 1):  # Show top 10
                    sym = s.get("symbol") or "N/A"
                    date = s.get("date") or "N/A"
                    src = s.get("source") or "unknown"
                    text_preview = s.get("text", "")[:150]
                    
                    st.write(f"**{i}.** [{src}] **{sym}** ({date})")
                    st.write(f"_{text_preview}..._")
                    st.divider()

# --- Footer with system info ---
with st.expander("System Information"):
    import datetime
    st.write(f"**Last Updated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**MCP Servers:** {len(servers)} configured")
    if st.session_state.get("startup_results"):
        successful = sum(st.session_state["startup_results"].values())
        total = len(st.session_state["startup_results"])
        st.write(f"**Server Status:** {successful}/{total} running")