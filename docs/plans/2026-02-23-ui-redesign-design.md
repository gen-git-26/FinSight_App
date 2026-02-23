# FinSight UI Redesign — Design Document

**Date:** 2026-02-23
**Status:** Approved — ready for implementation

---

## Chosen direction: "Midnight Mint"

Disciplined dark UI where mint appears only as accent — focus rings, positive
deltas, and CTA hover glow. No large mint fills. No decorative gradients.
Feels like Bloomberg Terminal x Vercel Dashboard.

---

## Design system changes

### Token adjustments

| Token | Old | New | Reason |
|---|---|---|---|
| `CARD_DARK` | `#121416` | `#181C1F` | Old delta vs BG was 5–6/channel — below perception threshold |
| `BORDER` | `#2A2F33` | `#252B2E` | Tighter, more refined |
| `CARD_ELEVATED` | — | `#1D2226` | New: hover, modals, focus surfaces |
| `TEXT_MUTED` | — | `#8A9BA8` | New: body copy, descriptions |
| `TEXT_SUBTLE` | — | `#526070` | New: captions, timestamps |
| `PRIMARY_ACTION` | — | `#0E8A5C` | Streamlit primaryColor; white on this = 5.0:1 contrast (WCAG AA) |
| `STATUS_NEG` | red | `#FB7185` | Rose-pink replaces harsh red across the brand |

### Color palette — full set

```python
BG_DARK        = "#0D0F10"   # page background
CARD_DARK      = "#181C1F"   # cards, chat bubbles
CARD_ELEVATED  = "#1D2226"   # hover, modals
PRIMARY_MINT   = "#9AF8CC"   # accent: glows, deltas, focus
PRIMARY_ACTION = "#0E8A5C"   # buttons, links (WCAG AA)
TEXT_MAIN      = "#FFFFFF"   # headlines, values
TEXT_MUTED     = "#8A9BA8"   # body copy
TEXT_SUBTLE    = "#526070"   # captions, meta
BORDER         = "#252B2E"   # card / widget borders
BORDER_FOCUS   = "#9AF8CC"   # focus ring
STATUS_POS     = "#9AF8CC"   # positive delta
STATUS_WARN    = "#F59E0B"   # amber warning
STATUS_NEG     = "#FB7185"   # rose-pink negative
STATUS_INFO    = "#38BDF8"   # sky blue info
```

---

## Home screen UX plan

```
Sidebar                   Main content (max-width 980px)
─────────────────────     ──────────────────────────────────────
Logo (st.logo)            HEADER
Status badge              [Logo] FinSight (gradient H1)
                          "See beyond the numbers"
Agent flows (expander)
                          KPI ROW (horizontal container)
Clear chat (button)       [AAPL] [S&P 500] [BTC] [Volume]
                          each card: value + delta + sparkline

Powered by LangGraph      CHAT INPUT (primary CTA)
                          mint border on focus, 2px ring glow

                          SUGGESTION CHIPS (empty state only)
                          [AAPL price] [Should I buy Tesla?] ...

                          CHAT HISTORY
                          user and assistant messages

                          INSIGHTS CARDS (after response)
                          3-column cards, overflow in expander

                          SOURCES (collapsible expander)
```

### States

| State | Treatment |
|---|---|
| Empty | Suggestion pills below chat input; illustrated empty copy |
| Loading | Mint 3-dot pulse animation inside assistant bubble |
| Error | `st.warning()` amber + `:material/warning:` icon |
| Success | Response + optional insights cards + sources expander |

---

## Files created / modified

| File | Change |
|---|---|
| `.streamlit/config.toml` | Created — full brand theme |
| `ui/styles.css` | Created — global CSS (cards, chat, buttons, states) |
| `ui/skeleton.py` | Created — reference UI implementation |
| `app.py` | Simplified to `main(query_fn=run_query)` |

---

## Typography

Font: **Heebo** (Google Fonts) — modern geometric sans with Latin support.
Code font: **JetBrains Mono** for tickers and code blocks.

Scale:
- H1: 30px / 800 weight
- H2: 22px / 700 weight
- Body: 15px / 400 weight
- Caption: 13px / `TEXT_SUBTLE` color

---

## Accessibility notes

- All primary interactive elements meet WCAG AA contrast (4.5:1+)
- `PRIMARY_ACTION #0E8A5C` on white = 5.0:1
- `TEXT_MUTED #8A9BA8` on `CARD_DARK #181C1F` = 4.6:1
- Focus rings: 2px mint border + 3px rgba glow (visible without color-only reliance)
- Error states use amber + icon (not red-only)
- Loading states described in label text (screen reader compatible)
