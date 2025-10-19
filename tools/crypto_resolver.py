# tools/crypto_resolver.py
"""
Dynamic Crypto Symbol Resolver
- Fetches from CoinGecko API (free, no auth needed)
- Caches locally for 7 days
- Supports fuzzy matching
- Falls back to hardcoded list if API fails
"""

import json
import os
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path


class CryptoResolver:
    """
    Smart crypto symbol resolver that:
    - Caches results locally (TTL: 7 days)
    - Fetches from CoinGecko API (free, no auth needed)
    - Falls back to hardcoded minimal list if API fails
    - Handles fuzzy matching
    """
    
    # Minimal fallback list (most popular cryptos)
    FALLBACK_CRYPTOS = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'SOL': 'Solana',
        'XRP': 'Ripple',
        'DOGE': 'Dogecoin',
        'LTC': 'Litecoin',
        'ADA': 'Cardano',
        'MATIC': 'Polygon',
        'AVAX': 'Avalanche',
        'LINK': 'Chainlink',
        'UNI': 'Uniswap',
        'AAVE': 'Aave',
        'ATOM': 'Cosmos',
        'NEAR': 'NEAR Protocol',
        'ARB': 'Arbitrum',
        'OP': 'Optimism',
        'APTOS': 'Aptos',
        'SUI': 'Sui',
        'XLM': 'Stellar',
        'ETC': 'Ethereum Classic',
        'BCH': 'Bitcoin Cash',
        'DOT': 'Polkadot',
        'ALGO': 'Algorand',
        'VET': 'VeChain',
        'ICP': 'Internet Computer',
    }
    
    def __init__(self, cache_dir: str = ".cache/crypto"):
        self.cache_dir = cache_dir
        self.cache_file = f"{cache_dir}/crypto_mapping.json"
        self._ensure_cache_dir()
        self._cache_data = self._load_cache()
        self._last_fetch = None
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> Dict[str, str]:
        """Load cache from disk (with TTL validation)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_time = data.get('timestamp', 0)
                    
                    # Check if cache is still valid (7 days TTL)
                    age_seconds = datetime.now().timestamp() - cached_time
                    if age_seconds < 7 * 24 * 3600:
                        cache_size = len(data.get('symbols', {}))
                        print(f"✅ [CryptoResolver] Loaded cache: {cache_size} symbols (age: {age_seconds/3600:.1f}h)")
                        return data.get('symbols', {})
                    else:
                        print(f"⚠️ [CryptoResolver] Cache expired (age: {age_seconds/3600:.1f}h)")
            except Exception as e:
                print(f"⚠️ [CryptoResolver] Cache read error: {e}")
        
        return {}
    
    def _save_cache(self, symbols: Dict[str, str]):
        """Save cache to disk."""
        try:
            data = {
                'timestamp': datetime.now().timestamp(),
                'symbols': symbols
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ [CryptoResolver] Cache saved: {len(symbols)} symbols")
        except Exception as e:
            print(f"❌ [CryptoResolver] Cache write error: {e}")
    
    async def _fetch_from_coingecko(self) -> Optional[Dict[str, str]]:
        """
        Fetch all crypto symbols from CoinGecko API.
        Returns mapping: {'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL', ...}
        """
        try:
            print("🔄 [CryptoResolver] Fetching from CoinGecko API...")
            async with httpx.AsyncClient(timeout=30) as client:
                # Get top 1000 cryptos
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/coins/list",
                    params={"order": "market_cap_desc", "per_page": 1000, "locale": "en"}
                )
                resp.raise_for_status()
                
                coins = resp.json()
                mapping = {}
                
                for coin in coins:
                    symbol = (coin.get('symbol') or '').upper()
                    name = (coin.get('name') or '').lower()
                    coin_id = (coin.get('id') or '').lower()
                    
                    if symbol:
                        # Map by symbol
                        mapping[symbol] = symbol
                        
                        # Map by name and ID (for fuzzy matching)
                        if name:
                            mapping[name] = symbol
                        if coin_id and coin_id != name:
                            mapping[coin_id] = symbol
                
                print(f"✅ [CryptoResolver] Fetched {len(coins)} cryptos, {len(mapping)} total mappings")
                self._last_fetch = datetime.now()
                return mapping
        
        except asyncio.TimeoutError:
            print("⏱️ [CryptoResolver] API timeout (30s)")
            return None
        except Exception as e:
            print(f"❌ [CryptoResolver] API fetch failed: {e}")
            return None
    
    def fetch_fresh(self) -> Dict[str, str]:
        """Synchronously fetch fresh data from CoinGecko."""
        try:
            fresh = asyncio.run(self._fetch_from_coingecko())
            if fresh:
                self._cache_data = fresh
                self._save_cache(fresh)
                return fresh
        except Exception as e:
            print(f"❌ [CryptoResolver] Fresh fetch failed: {e}")
        
        return {}
    
    def resolve(self, query_text: str, use_cache_only: bool = True) -> Dict[str, Any]:
        """
        Resolve crypto symbol from query text.
        
        Args:
            query_text: User input (e.g., "solana price", "what is bitcoin?")
            use_cache_only: If True, don't fetch fresh (faster)
        
        Returns:
            {
                'symbol': 'SOL',
                'name': 'Solana',
                'confidence': 0.95,
                'found': True,
                'source': 'cache'  # or 'api', 'fallback'
            }
        """
        if not query_text:
            return {
                'symbol': None,
                'name': None,
                'confidence': 0.0,
                'found': False,
                'source': 'none'
            }
        
        query_lower = query_text.lower()
        
        # Try cache first
        if self._cache_data:
            for key, symbol in self._cache_data.items():
                if key.lower() in query_lower:
                    return {
                        'symbol': symbol,
                        'name': key.title(),
                        'confidence': 0.95,
                        'found': True,
                        'source': 'cache'
                    }
        
        # If cache is empty and allowed, fetch fresh
        if not self._cache_data and not use_cache_only:
            fresh = self.fetch_fresh()
            for key, symbol in fresh.items():
                if key.lower() in query_lower:
                    return {
                        'symbol': symbol,
                        'name': key.title(),
                        'confidence': 0.95,
                        'found': True,
                        'source': 'api'
                    }
        
        # Try fallback list
        for key, name in self.FALLBACK_CRYPTOS.items():
            if key.lower() in query_lower or name.lower() in query_lower:
                return {
                    'symbol': key,
                    'name': name,
                    'confidence': 0.85,
                    'found': True,
                    'source': 'fallback'
                }
        
        # Not found
        return {
            'symbol': None,
            'name': None,
            'confidence': 0.0,
            'found': False,
            'source': 'none'
        }
    
    def resolve_symbol_only(self, symbol_or_name: str) -> Optional[str]:
        """Get standard symbol for a crypto (fast lookup)."""
        result = self.resolve(symbol_or_name, use_cache_only=True)
        return result['symbol'] if result['found'] else None
    
    def is_crypto(self, text: str) -> bool:
        """Check if text contains a crypto reference."""
        result = self.resolve(text, use_cache_only=True)
        return result['found']


# ============================================================================
# Global instance (singleton pattern)
# ============================================================================

_crypto_resolver: Optional[CryptoResolver] = None

def get_crypto_resolver() -> CryptoResolver:
    """Get or create global resolver instance."""
    global _crypto_resolver
    if _crypto_resolver is None:
        _crypto_resolver = CryptoResolver()
    return _crypto_resolver


def resolve_crypto(query: str, use_cache_only: bool = True) -> Dict[str, Any]:
    """Convenience function."""
    return get_crypto_resolver().resolve(query, use_cache_only)


def is_crypto_query(text: str) -> bool:
    """Check if query mentions a crypto."""
    return get_crypto_resolver().is_crypto(text)