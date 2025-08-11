from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import asyncpg
from dotenv import load_dotenv


async def get_db_pool() -> asyncpg.Pool:
    load_dotenv()
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        raise RuntimeError("SUPABASE_DB_URL must be set in .env, e.g. postgres://user:pass@host:5432/db")
    return await asyncpg.create_pool(dsn, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "10")))


UPSERT_SQL = """
insert into market_data.ohlcv (
  bucket_start, interval, symbol, open, high, low, close, volume, trades, vwap
) values (
  $1::timestamptz, $2::text, $3::text, $4::numeric, $5::numeric, $6::numeric, $7::numeric, $8::numeric, $9::bigint, $10::numeric
)
on conflict (bucket_start, interval, symbol) do update set
  open   = excluded.open,
  high   = greatest(ohlcv.high, excluded.high),
  low    = least(ohlcv.low, excluded.low),
  close  = excluded.close,
  volume = ohlcv.volume + excluded.volume,
  trades = ohlcv.trades + excluded.trades,
  vwap   = excluded.vwap
"""


async def upsert_ohlcv_rows(pool: asyncpg.Pool, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                UPSERT_SQL,
                [
                    (
                        r["bucket_start"],
                        r["interval"],
                        r["symbol"],
                        r["open"],
                        r["high"],
                        r["low"],
                        r["close"],
                        r["volume"],
                        r.get("trades", 0),
                        r.get("vwap"),
                    )
                    for r in rows
                ],
            )
