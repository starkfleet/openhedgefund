from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Union

from dateutil import parser as dateparser

from .client import create_spot_client


BinanceInterval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]


_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}


def _to_unix_ms(dt: Union[str, datetime]) -> int:
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp() * 1000)
    parsed = dateparser.parse(dt)
    if parsed is None:
        raise ValueError(f"Unable to parse date: {dt}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def _normalize_symbol(symbol: str) -> str:
    key = symbol.upper()
    return _SYMBOLS.get(key, key)


def get_historical_ohlcv(
    symbol: str,
    interval: BinanceInterval,
    start: Union[str, datetime],
    end: Union[str, datetime],
    limit_per_request: int = 1000,
) -> List[Dict]:
    """Fetch historical OHLCV for a symbol within [start, end]."""
    client = create_spot_client()

    sym = _normalize_symbol(symbol)
    start_ms = _to_unix_ms(start)
    end_ms = _to_unix_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    all_rows: List[Dict] = []
    cursor = start_ms

    while cursor < end_ms:
        klines = client.klines(
            symbol=sym,
            interval=interval,
            startTime=cursor,
            endTime=end_ms,
            limit=limit_per_request,
        )
        if not klines:
            break

        for k in klines:
            open_time = int(k[0])
            close_time = int(k[6])
            if open_time >= end_ms:
                break
            all_rows.append(
                {
                    "symbol": sym,
                    "interval": interval,
                    "open_time": open_time,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": close_time,
                    "trades": int(k[8]),
                }
            )

        last_close = int(klines[-1][6])
        next_cursor = last_close + 1
        if next_cursor <= cursor:
            next_cursor = cursor + 1
        cursor = next_cursor

    return all_rows


def get_btc_ohlcv(
    interval: BinanceInterval,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> List[Dict]:
    return get_historical_ohlcv("BTC", interval, start, end)


def get_eth_ohlcv(
    interval: BinanceInterval,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> List[Dict]:
    return get_historical_ohlcv("ETH", interval, start, end)


def get_sol_ohlcv(
    interval: BinanceInterval,
    start: Union[str, datetime],
    end: Union[str, datetime],
) -> List[Dict]:
    return get_historical_ohlcv("SOL", interval, start, end)
