# ui/skeleton.py
"""
FinSight UI — production implementation.

Layout: branded header + sidebar nav + full-screen chat.
No KPI graphs. All states handled (loading / error / empty / active).

Usage:
    streamlit run ui/skeleton.py              # standalone demo
    from ui.skeleton import main; main(query_fn)  # integrate with agent
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens — mirrors .streamlit/config.toml and ui/styles.css
# ---------------------------------------------------------------------------
LOGO_PATH     = "ui/logo.png"
BOT_ICON_PATH = "ui/bot_icon.png"

PRIMARY_MINT = "#9AF8CC"
TEXT_MAIN    = "#FFFFFF"
TEXT_MUTED   = "#8A9BA8"
TEXT_SUBTLE  = "#526070"

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSight",
    page_icon=BOT_ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    """Load ui/styles.css and inject into the page."""
    css_path = Path("ui/styles.css")
    if css_path.exists():
        st.html(f"<style>{css_path.read_text(encoding='utf-8')}</style>")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _clear_chat() -> None:
    st.session_state["messages"] = []


def render_sidebar() -> None:
    """Sidebar: logo via st.logo, status, agent info, clear chat."""
    with st.sidebar:
        # st.logo places the image in the sidebar header area
        st.logo(LOGO_PATH, size="large")

        st.caption("AI-powered financial analysis")

        st.space("medium")
        st.badge("Connected", icon=":material/check_circle:", color="green")
        st.space("medium")

        with st.expander(":material/account_tree: Agent flows"):
            st.markdown(
                f"""
                <div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.9;'>
                  <b style='color:{TEXT_MAIN};'>Standard flow</b><br>
                  Router &rarr; Fetcher &rarr; Analyst &rarr; Composer
                  <br><br>
                  <b style='color:{TEXT_MAIN};'>Full trading flow (A2A)</b><br>
                  4 Analysts &rarr; Researchers &rarr; Trader
                  &rarr; Risk Manager &rarr; Fund Manager
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.space("small")

        st.button(
            ":material/delete_sweep: Clear chat",
            use_container_width=True,
            on_click=_clear_chat,
        )

        st.space("large")
        st.caption("Powered by LangGraph & GPT-4o")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header() -> None:
    """
    Branded header: logo on the left (fixed 72px), title + tagline on right.
    A thin border-bottom separates it from the chat area.
    """
    st.html("<div class='fs-header'>")

    col_logo, col_text = st.columns([0.16, 0.84], vertical_alignment="center")

    with col_logo:
        st.image(LOGO_PATH, width=160)

    with col_text:
        st.markdown(
            f"""
            <div style='line-height:1.15; padding-left: 0.5rem;'>
              <span style='
                font-size: 2.8rem;
                font-weight: 900;
                background: linear-gradient(115deg, {TEXT_MAIN} 50%, {PRIMARY_MINT});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -1px;
                display: block;
              '>FinSight</span>
              <span style='
                display: block;
                font-size: 1.05rem;
                color: {TEXT_MUTED};
                font-weight: 400;
                margin-top: 4px;
                letter-spacing: 0.1px;
              '>See beyond the numbers</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.html("</div>")


# ---------------------------------------------------------------------------
# Welcome / empty state
# ---------------------------------------------------------------------------
def render_welcome() -> None:
    """Centered welcome message shown before the first message."""
    st.html(
        """
        <div class="fs-welcome">
          <p class="fs-welcome__title">How can I help you today?</p>
          <p class="fs-welcome__sub">
            Ask me about stock prices, trading decisions, crypto,
            options analysis, or company fundamentals.
          </p>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Suggestion chips
# ---------------------------------------------------------------------------
_SUGGESTIONS: dict[str, str] = {
    "AAPL price":          "What is Apple's current stock price?",
    "Should I buy Tesla?": "Should I buy Tesla stock right now?",
    "Bitcoin analysis":    "Give me a full analysis of Bitcoin.",
    "NVIDIA news":         "What is the latest news on NVIDIA?",
}


def render_suggestion_chips() -> str | None:
    """Clickable suggestion pills. Returns the resolved prompt or None."""
    chosen = st.pills(
        "Try asking:",
        list(_SUGGESTIONS.keys()),
        label_visibility="visible",
    )
    return _SUGGESTIONS.get(chosen) if chosen else None


# ---------------------------------------------------------------------------
# Loading state
# ---------------------------------------------------------------------------
def render_loading(label: str = "Analyzing...") -> None:
    """Mint three-dot pulse shown inside the assistant bubble."""
    st.html(
        f"""
        <div class="fs-loading">
          <div class="fs-loading__dots">
            <span></span><span></span><span></span>
          </div>
          <p class="fs-loading__label">{label}</p>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Error state
# ---------------------------------------------------------------------------
def render_error(message: str) -> None:
    """Amber warning — no red, icon-backed."""
    st.warning(f"Something went wrong: {message}", icon=":material/warning:")


# ---------------------------------------------------------------------------
# Insights cards
# ---------------------------------------------------------------------------
def render_insights(insights: list[dict] | None) -> None:
    """
    Render analyst insight cards below a response.

    Each dict: {title: str, summary: str, badge_color: str}
    badge_color: "green" | "blue" | "orange" | "violet" | "gray"
    """
    if not insights:
        return

    st.markdown("#### Insights")

    visible  = insights[:3]
    overflow = insights[3:]

    cols = st.columns(len(visible))
    for col, item in zip(cols, visible):
        with col:
            with st.container(border=True):
                st.badge(item["title"], color=item.get("badge_color", "blue"))
                st.markdown(
                    f"<p style='color:{TEXT_MUTED}; font-size:0.875rem; "
                    f"margin:0.5rem 0 0;'>{item['summary']}</p>",
                    unsafe_allow_html=True,
                )

    if overflow:
        with st.expander(
            f"{len(overflow)} more insights",
            icon=":material/analytics:",
        ):
            for item in overflow:
                st.markdown(f"**{item['title']}** — {item['summary']}")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def render_sources(sources: list[str] | None) -> None:
    """Collapsible sources list below a response."""
    if not sources:
        return

    with st.expander("Sources", icon=":material/link:", expanded=False):
        for src in sources:
            st.markdown(
                f"<p style='color:{TEXT_MUTED}; font-size:0.85rem;'>"
                f"- {src}</p>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
def render_chat_history() -> None:
    """Replay all stored messages with correct avatars."""
    for msg in st.session_state.get("messages", []):
        role   = msg["role"]
        avatar = BOT_ICON_PATH if role == "assistant" else None

        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

            if role == "assistant":
                render_insights(msg.get("insights"))
                render_sources(msg.get("sources"))


# ---------------------------------------------------------------------------
# Chat input + agent call
# ---------------------------------------------------------------------------
def handle_chat_input(query_fn) -> None:
    """
    Render the chat input bar, handle suggestion chips on first visit,
    call query_fn, and manage loading / error states.

    query_fn signature: (prompt: str, user_id: str) -> str
    """
    prompt: str | None = st.chat_input(
        "Ask about stocks, crypto, options, or fundamentals..."
    )

    # Suggestion chips only before the first message
    if not st.session_state.get("messages"):
        suggestion = render_suggestion_chips()
        if suggestion:
            prompt = suggestion

    if not prompt:
        return

    # Store and render user message
    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant bubble — loading → result / error
    with st.chat_message("assistant", avatar=BOT_ICON_PATH):
        loading_slot = st.empty()
        with loading_slot.container():
            render_loading("Analyzing your question...")

        try:
            result = query_fn(
                prompt,
                user_id=st.session_state.get("user_id", "default"),
            )
            loading_slot.empty()
            st.markdown(result)

            st.session_state["messages"].append(
                {"role": "assistant", "content": result}
            )

        except Exception as exc:
            loading_slot.empty()
            render_error(str(exc))
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"Error: {exc}"}
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(query_fn=None) -> None:
    """
    Render the full FinSight UI.

    Args:
        query_fn: Callable[[str, str], str] — agent query function.
                  If None, a stub is used for standalone demo.
    """
    if query_fn is None:
        def query_fn(prompt: str, user_id: str) -> str:  # noqa: E306
            time.sleep(1.0)
            return (
                f"*[Demo mode]* Received: **{prompt}**\n\n"
                "Pass `query_fn=run_query` to connect the real agent."
            )

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user_id", "default")

    _inject_css()
    render_sidebar()
    render_header()

    # Show welcome screen only before the first message
    if not st.session_state["messages"]:
        render_welcome()

    render_chat_history()
    handle_chat_input(query_fn)


if __name__ == "__main__":
    main()
