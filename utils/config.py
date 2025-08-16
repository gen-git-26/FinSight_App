
import os
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().strip('"').strip("'").lower()
    return v in ("1", "true", "yes", "on")



class Settings(BaseModel):
    openai_api_key: str
    openai_model: str
    openai_embed_model: str

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_memory_collection: str

    enable_yahoo: bool
    enable_alpha_vantage: bool
    enable_coinstats: bool

    yahoo_mcp_dir: str = "" 

    alphavantage_api_key: str
    finnhub_api_key: str
    coinstats_api_key: str

    watchlist: List[str]
    update_interval_min: int
    news_provider: str
    news_window_days: int

def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "finsight_docs"),
        qdrant_memory_collection=os.getenv("QDRANT_MEMORY_COLLECTION", "finsight_memory"),
        enable_yahoo=_get_bool("MCP_ENABLE_YAHOO", True),
        enable_alpha_vantage=_get_bool("MCP_ENABLE_ALPHA_VANTAGE", True),
        enable_coinstats=_get_bool("MCP_ENABLE_COINSTATS", True),
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        coinstats_api_key=os.getenv("COINSTATS_API_KEY", ""),
        alpaca_key=os.getenv("ALPACA_KEY", ""),
        alpaca_secret=os.getenv("ALPACA_SECRET", ""),
    )


