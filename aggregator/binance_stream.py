from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Tuple

from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient as WSClient
from dotenv import load_dotenv

INTERVALS = {
    "1s": 1,
    "1m": 60,
}

Symbol = str
Interval = Literal["1s", "1m"]


@dataclass
class BarState:
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


def floor_to_interval(ts: datetime, seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


class Aggregator:
    def __init__(self, symbols: List[Symbol], intervals: List[Interval]) -> None:
        self.symbols = symbols
        self.intervals = intervals
        self.state: Dict[Tuple[Symbol, Interval, datetime], BarState] = {}

    def process_tick(self, symbol: Symbol, price: float, qty: float, ts: datetime) -> List[Dict]:
        finalized: List[Dict] = []
        for interval in self.intervals:
            window_s = INTERVALS[interval]
            bucket = floor_to_interval(ts, window_s)
            key = (symbol, interval, bucket)
            bar = self.state.get(key)
            if bar is None:
                bar = BarState(open=price, high=price, low=price, close=price, volume=0.0, trades=0)
                self.state[key] = bar
            bar.close = price
            if price > bar.high:
                bar.high = price
            if price < bar.low:
                bar.low = price
            bar.volume += qty
            bar.trades += 1

        # finalize bars for buckets older than current - interval
        watermark = ts - timedelta(seconds=max(INTERVALS[i] for i in self.intervals))
        to_finalize = [k for k in self.state.keys() if k[2] <= watermark]
        for key in sorted(to_finalize, key=lambda x: x[2]):
            sym, interval, bucket = key
            bar = self.state.pop(key)
            finalized.append(
                {
                    "bucket_start": bucket.isoformat(),
                    "interval": interval,
                    "symbol": sym,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "trades": bar.trades,
                }
            )
        return finalized


async def run_stream(symbols: List[str], handle_rows):
    load_dotenv()

    agg = Aggregator(symbols=symbols, intervals=["1s", "1m"])
    queue: asyncio.Queue[List[Dict]] = asyncio.Queue(maxsize=1000)

    def on_message(_, message: str):
        data = json.loads(message)
        if "p" in data and "q" in data and "s" in data and "T" in data:
            price = float(data["p"])  # trade price
            qty = float(data["q"])    # trade qty
            symbol = data["s"]
            ts = datetime.fromtimestamp(int(data["T"]) / 1000.0, tz=timezone.utc)
            finalized = agg.process_tick(symbol, price, qty, ts)
            if finalized:
                try:
                    queue.put_nowait(finalized)
                except asyncio.QueueFull:
                    # drop on overload; adjust maxsize as needed
                    pass

    # create WS client
    ws_client = WSClient(on_message=on_message)

    try:
        for sym in symbols:
            ws_client.agg_trade(symbol=sym.lower())
        # consumer loop for DB writes
        while True:
            rows = await queue.get()
            try:
                await handle_rows(rows)
            finally:
                queue.task_done()
    finally:
        ws_client.stop()
