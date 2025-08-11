from __future__ import annotations

import os
from typing import Tuple

from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env file if present."""
    load_dotenv()


def get_binance_credentials() -> Tuple[str, str]:
    """Return (api_key, api_secret) from environment.

    Requires BINANCE_KEY and BINANCE_SEC to be set in the environment or .env file.
    """
    load_env()
    api_key = os.getenv("BINANCE_KEY") or ""
    api_secret = os.getenv("BINANCE_SEC") or ""
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_KEY and BINANCE_SEC must be set in .env")
    return api_key, api_secret
