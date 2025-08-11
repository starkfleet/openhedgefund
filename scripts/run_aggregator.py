from __future__ import annotations

import asyncio
import os
from typing import List

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aggregator.binance_stream import run_stream
from aggregator.supabase_writer import get_db_pool, upsert_ohlcv_rows


async def main() -> None:
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    pool = await get_db_pool()

    async def handle_rows(rows):
        await upsert_ohlcv_rows(pool, rows)
        print(f"Upserted {len(rows)} rows")

    await run_stream(symbols, handle_rows)


if __name__ == "__main__":
    asyncio.run(main())
