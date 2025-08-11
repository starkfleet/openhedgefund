## System Design for Low-latency OHLC Aggregation of High-frequency Crypto Streams

Designing a scalable, low-latency system for generating OHLC (Open, High, Low, Close) bars, or candlestick data, from a high-frequency cryptocurrency data stream involves several key architectural choices. Below is the updated plan for a Binance-to-DB data layer using Supabase Postgres (no TimescaleDB) and a custom Python stream aggregator.

### 1. **Data Ingestion & Streaming Layer**
- **Exchange Source (Binance)**: Use Binance WebSocket streams for real-time trades/aggTrade and order book updates. Supplement with Binance REST for backfill and recovery on disconnects.[1][2]
- **Lightweight Buffering**: Primarily in-memory queues within the custom Python aggregator for minimal latency. Optionally add Redis Streams/Kafka later if we fan-out to multiple downstream consumers.

### 2. **Real-time OHLC Aggregator (Custom Python)**
- **Architecture**: A custom Python service (asyncio) maintains in-memory state per `(symbol, interval)` using tumbling windows. One task per stream/symbol; a scheduler finalizes bars at interval boundaries.
- **Aggregation Algorithm**: O(1) updates per tick: store open at window start, update high/low, keep running volume, and set close from the latest tick.[2][3]
- **Intervals**: Start with 1s and 1m; extend to 5m/15m/1h as needed. Multiple intervals can be updated from the same tick.
- **Fault Tolerance**:
  - Persist a write-ahead watermark per `(symbol, interval)` in Postgres to resume after restarts.
  - Use idempotent upserts to avoid duplicates.
  - On reconnect, backfill from Binance REST between `last_watermark_ts` and `now`.
- **Performance**: Prefer `asyncio` websockets and batched inserts per 50–250 rows or per 25–100 ms, whichever comes first. Consider `uvloop` and `asyncpg` for low-latency Postgres I/O.

### 3. **Data Storage: Supabase (Postgres only)**
- **Supabase**: Managed Postgres with integrated APIs, auth, and storage. No TimescaleDB. We will use standard Postgres tables, indexes, and optionally native table partitioning.
- **Schema**:
  - `market_data.ticks` table for optional raw trade ticks
  - `market_data.ohlcv` table for finalized bars from the Python aggregator
  - Primary key `(bucket_start, interval, symbol)` ensures idempotent writes.

Recommended DDL (run in Supabase SQL editor):

```sql
-- Namespacing
create schema if not exists market_data;

-- Optional raw ticks for audits/backfill
create table if not exists market_data.ticks (
  ts timestamptz not null,
  symbol text not null,
  price numeric(38, 18) not null,
  quantity numeric(38, 18) not null,
  trade_id bigint,
  is_buyer_maker boolean,
  source text default 'binance',
  primary key (ts, symbol, trade_id)
);
create index if not exists ticks_symbol_ts_idx on market_data.ticks (symbol, ts desc);

-- Finalized OHLCV bars from the aggregator
create table if not exists market_data.ohlcv (
  bucket_start timestamptz not null,
  interval text not null, -- e.g., '1s','1m','5m','1h'
  symbol text not null,
  open numeric(38, 18) not null,
  high numeric(38, 18) not null,
  low numeric(38, 18) not null,
  close numeric(38, 18) not null,
  volume numeric(38, 18) not null,
  trades bigint default 0,
  vwap numeric(38, 18),
  source text default 'binance',
  primary key (bucket_start, interval, symbol)
);
create index if not exists ohlcv_symbol_interval_ts_idx on market_data.ohlcv (symbol, interval, bucket_start desc);

-- (Optional) Native partitioning for large datasets (example: monthly range partitions)
-- 1) Convert to partitioned table (run only on a new table)
-- alter table market_data.ohlcv partition by range (bucket_start);
-- 2) Create partitions as needed
-- create table if not exists market_data.ohlcv_2024_07 partition of market_data.ohlcv for values from ('2024-07-01') to ('2024-08-01');
```

- **Access Path**:
  - Aggregator writes directly via Postgres connection string (service role key) for best latency and upserts.
  - BI/analytics can query via Supabase SQL, PostgREST, or RPC.

| Component           | Tech Options                                 | Why?                               |
|---------------------|-----------------------------------------------|------------------------------------|
| Ingestion           | Binance WS/REST; optional Redis/Kafka         | Reliable, high-throughput          |
| Stream Aggregator   | Custom Python (asyncio, asyncpg)              | Real-time, low-latency, flexible   |
| OHLC Storage        | Supabase Postgres                             | Managed Postgres, simple & robust |

### 4. **Low-latency Principles**
- **Minimize Network Round-trips** via persistent WS streams and batched DB writes.[7][8]
- **Optimize I/O** with async Postgres drivers (`asyncpg`), statement reuse, and connection pooling.[9][7]
- **Horizontal Scalability**: Partition by symbol and interval; run multiple aggregator replicas with sharding.
- **Fault Tolerance**: Idempotent upserts on `(bucket_start, interval, symbol)`; maintain watermarks; automatic WS reconnect and REST backfill.[1][10]

### 5. **Example Real-time Calculation Flow**
1. Tick arrives from Binance WS and enters the Python aggregator’s in-memory queue.
2. Aggregator updates current OHLC for all configured intervals in O(1).
3. At interval end, bar is finalized and upserted into `market_data.ohlcv` (Postgres table) on Supabase.
4. Optional: raw ticks are inserted into `market_data.ticks` for audit/backfill.
5. Client/UI/BI fetches completed bars via SQL/PostgREST or uses continuous aggregates.

Example upsert from the aggregator:

```sql
insert into market_data.ohlcv (
  bucket_start, interval, symbol, open, high, low, close, volume, trades, vwap
) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
on conflict (bucket_start, interval, symbol) do update set
  open = excluded.open,
  high = greatest(ohlcv.high, excluded.high),
  low  = least(ohlcv.low, excluded.low),
  close = excluded.close,
  volume = ohlcv.volume + excluded.volume,
  trades = ohlcv.trades + excluded.trades,
  vwap = excluded.vwap;
```

- **Supabase Setup Checklist**:
  - Create project and database
  - Run the DDL above; verify tables and indexes
  - Configure a service role key for the aggregator; consider disabling RLS on `market_data.*` or create proper policies for service role
  - Configure connection pooling (pgBouncer) and max connections
  - (Optional) Use native range partitioning for `market_data.ohlcv` for very large datasets
- **Backfill**: On deployment or gaps, fetch from Binance REST to reconstruct missing bars and insert via the same upsert path.
- **Resampling & Cleaning**: Fill missing intervals, keep UTC timestamps, and handle outliers as needed.[11][2]
- **Monitoring**: Use Supabase metrics + Prometheus/Grafana; add DB latency/error logs and WS reconnect metrics.[10]

***

**In summary:**
Build the Binance-to-DB pipeline with a custom Python aggregator consuming Binance WS, writing finalized bars to Supabase Postgres tables (no TimescaleDB). Use idempotent upserts, async I/O, optional raw tick storage, and standard Postgres features (indexes, partitioning) for efficient storage and fast queries. Focus on batching, parallelism, and low-latency I/O.[6][4][7]

### 7. Implementation Path (Repository Layout)
- `binance_data/`:
  - `config.py`: Load `.env` and provide `get_binance_credentials()`
  - `client.py`: Create authenticated Binance Spot client
  - `ohlcv.py`: Historical OHLC fetchers with pagination for BTC/ETH/SOL
- `scripts/fetch_ohlcv.py`: CLI to fetch historical bars (start/end date, interval) using the module above
- `requirements.txt`: Python dependencies (`binance-connector`, `python-dotenv`, `python-dateutil`)
- Next steps: integrate writer to Supabase (`asyncpg`) and wire into aggregator.

[1] https://docs.dolphindb.com/en/Tutorials/integrating_crypto_market_data.html
[2] https://docs.dolphindb.com/en/Tutorials/OHLC.html
[3] https://hirzels.com/martin/papers/debs17-daba.pdf
[4] https://www.tigerdata.com/learn/the-best-time-series-databases-compared
[5] https://kite.trade/forum/discussion/2078/suggest-me-a-database-to-store-tick-data
[6] https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-022-00323-4
[7] https://www.geeksforgeeks.org/system-design/low-latency-design-patterns/
[8] https://www.linkedin.com/pulse/large-scale-low-latency-system-design-journey-towards-nishant-kumar
[9] https://dzone.com/articles/real-time-market-data-processing-designing-systems
[10] https://aws.amazon.com/blogs/database/build-a-near-real-time-data-aggregation-pipeline-using-a-serverless-event-driven-architecture/
[11] https://finage.co.uk/blog/how-to-use-ohlcv-data-to-improve-technical-analysis-in-trading--684007623458598454e3dd10
[12] https://www.sciencedirect.com/science/article/pii/S0957417423023084
[13] https://papers.ssrn.com/sol3/Delivery.cfm/4834362.pdf?abstractid=4834362
[14] https://arxiv.org/html/2406.14537v1
[15] https://discovery.ucl.ac.uk/id/eprint/10155501/2/AndrewDMannPhDFinal.pdf
[16] https://docs.coinapi.io/market-data/how-to-guides/get-historical-ohlcv-data-using-coinapi
[17] https://www.reddit.com/r/algotrading/comments/1do5ulr/need_advice_integrating_realtime_and_historical/
[18] https://github.com/topics/ohlcv-data