from __future__ import annotations

from binance.spot import Spot

from .config import get_binance_credentials


def create_spot_client() -> Spot:
    """Create an authenticated Binance Spot client using credentials from .env."""
    api_key, api_secret = get_binance_credentials()
    return Spot(api_key=api_key, api_secret=api_secret)
