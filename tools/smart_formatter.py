# tools/smart_formatter.py
"""
Smart output formatter that detects content type and formats accordingly.
Handles tables, dicts, lists, and text intelligently.
"""

from typing import Any, Dict, List, Tuple
import json
import pandas as pd


def _detect_content_type(parsed: Any) -> str:
    """
    Detect the type of content for optimal display.
    Returns: "table", "dict", "text", "error", "unknown"
    """
    if parsed is None:
        return "empty"
    
    # List of dicts with many records = TABLE
    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            # Check if it's a real data table (multiple columns)
            first_item = parsed[0]
            if len(first_item) >= 3:  # At least 3 columns
                return "table"
            # Few columns - might be time series
            if len(parsed) > 5 and all(isinstance(item, dict) for item in parsed[:5]):
                return "table"
        # List of simple values
        return "list"
    
    # Single dict with many fields = STRUCTURED DATA (dict view)
    if isinstance(parsed, dict):
        if len(parsed) > 5:
            return "dict_expanded"
        return "dict_compact"
    
    # Numbers, strings = TEXT
    if isinstance(parsed, (int, float, str)):
        return "text"
    
    return "unknown"


def _format_as_table(data: List[Dict], max_rows: int = 100) -> Tuple[str, bool]:
    """Format as pandas DataFrame for Streamlit (return JSON for serialization)."""
    try:
        df = pd.DataFrame(data[:max_rows])
        
        # Format numeric columns nicely but keep as numbers for JSON
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Don't convert to string - keep numeric for dataframe
                pass
        
        # Return JSON serializable format for Streamlit
        return df.to_json(orient='records'), True
    except Exception as e:
        print(f"[smart_formatter] Table formatting error: {e}")
        return json.dumps(data), False


def _format_as_dict(data: Dict, max_fields: int = 50) -> str:
    """Format dict with nice key-value display in markdown."""
    lines = []
    
    # Priority fields to show first
    priority_fields = [
        'currentPrice', 'regularMarketPrice', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
        'marketCap', 'volume', 'pe', 'eps', 'dividend', 'yield',
        'shortName', 'sector', 'industry', 'beta', 'price', 'priceUsd'
    ]
    
    shown = set()
    
    # Show priority fields first
    for field in priority_fields:
        if field in data:
            value = data[field]
            formatted_val = _format_value(value)
            lines.append(f"• **{field}**: {formatted_val}")
            shown.add(field)
    
    # Show remaining fields
    for field, value in sorted(data.items())[:max_fields]:
        if field not in shown and not field.startswith('_'):
            formatted_val = _format_value(value)
            lines.append(f"• **{field}**: {formatted_val}")
            shown.add(field)
    
    if len(data) > len(shown):
        lines.append(f"\n_(... and {len(data) - len(shown)} more fields)_")
    
    return "\n".join(lines)


def _format_value(val: Any) -> str:
    """Format a single value nicely for display."""
    if val is None or val == "":
        return "N/A"
    
    if isinstance(val, bool):
        return "✅ Yes" if val else "❌ No"
    
    if isinstance(val, float):
        # Large numbers (market cap, volume)
        if abs(val) > 1_000_000:
            return f"{val/1_000_000:.2f}M"
        elif abs(val) > 1_000:
            return f"{val/1_000:.2f}K"
        # Small numbers (prices)
        elif 0 < abs(val) < 0.01:
            return f"{val:.6f}"
        # Regular prices/percentages
        else:
            return f"{val:,.2f}"
    
    if isinstance(val, int):
        return f"{val:,}" if abs(val) > 999 else str(val)
    
    return str(val)


def _format_data_as_text(parsed: Any, content_type: str = None) -> Tuple[str, str, bool]:
    """
    Format data intelligently based on content type.
    
    Returns:
        (formatted_text, content_type, is_dataframe)
        - formatted_text: display text (markdown or JSON)
        - content_type: "table", "dict", "text", etc.
        - is_dataframe: True if should use st.dataframe()
    """
    if content_type is None:
        content_type = _detect_content_type(parsed)
    
    if content_type == "empty":
        return "No data returned", "empty", False
    
    if content_type == "table":
        df_json, is_df = _format_as_table(parsed)
        return df_json, "table", True
    
    if content_type in ("dict_expanded", "dict_compact"):
        text = _format_as_dict(parsed)
        return text, "dict", False
    
    if content_type == "list":
        text = "\n".join([f"• {_format_value(item)}" for item in parsed[:20]])
        if len(parsed) > 20:
            text += f"\n\n_(... and {len(parsed) - 20} more items)_"
        return text, "list", False
    
    if content_type == "text":
        return _format_value(parsed), "text", False
    
    return json.dumps(parsed, ensure_ascii=False, indent=2), "unknown", False