
-- Create market_data schema
CREATE SCHEMA IF NOT EXISTS market_data;

-- Raw trade ticks table (optional, for audit/backfill)
CREATE TABLE IF NOT EXISTS market_data.ticks (
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  price NUMERIC(38, 18) NOT NULL,
  quantity NUMERIC(38, 18) NOT NULL,
  trade_id BIGINT,
  is_buyer_maker BOOLEAN,
  source TEXT DEFAULT 'binance',
  PRIMARY KEY (ts, symbol, trade_id)
);

-- Index for fast symbol + time queries
CREATE INDEX IF NOT EXISTS ticks_symbol_ts_idx 
ON market_data.ticks (symbol, ts DESC);

-- OHLCV bars table (main data from aggregator)
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
  bucket_start TIMESTAMPTZ NOT NULL,
  interval TEXT NOT NULL, -- '1s', '1m', '5m', '1h'
  symbol TEXT NOT NULL,
  open NUMERIC(38, 18) NOT NULL,
  high NUMERIC(38, 18) NOT NULL,
  low NUMERIC(38, 18) NOT NULL,
  close NUMERIC(38, 18) NOT NULL,
  volume NUMERIC(38, 18) NOT NULL,
  trades BIGINT DEFAULT 0,
  vwap NUMERIC(38, 18),
  source TEXT DEFAULT 'binance',
  PRIMARY KEY (bucket_start, interval, symbol)
);

-- Index for fast symbol + interval + time queries
CREATE INDEX IF NOT EXISTS ohlcv_symbol_interval_ts_idx 
ON market_data.ohlcv (symbol, interval, bucket_start DESC);

-- Optional: Add native partitioning or maintenance jobs here if needed
