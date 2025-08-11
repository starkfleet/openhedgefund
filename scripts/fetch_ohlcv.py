from __future__ import annotations

import argparse
import json

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from binance_data.ohlcv import get_historical_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV from Binance")
    parser.add_argument("symbol", help="Symbol alias or pair, e.g. BTC, ETH, SOL, BTCUSDT")
    parser.add_argument("interval", help="Binance interval, e.g. 1m, 5m, 1h, 1d")
    parser.add_argument("start", help="Start datetime (ISO8601)")
    parser.add_argument("end", help="End datetime (ISO8601)")
    parser.add_argument("--limit", type=int, default=1000, help="Limit per request (default 1000)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    rows = get_historical_ohlcv(args.symbol, args.interval, args.start, args.end, args.limit)
    if args.pretty:
        print(json.dumps(rows, indent=2))
    else:
        print(json.dumps(rows))


if __name__ == "__main__":
    main()
