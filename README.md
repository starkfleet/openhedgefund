# openhedgefund
Open source agentic hedge fund

<img width="1530" height="1003" alt="Screenshot 2025-08-14 at 1 09 04 PM" src="https://github.com/user-attachments/assets/86c420b8-7d1d-4ed2-a381-9025f479265f" />

test input.
```
{
  "buy_condition": "ema(10) > ema(20)",
  "sell_condition": "ema(20) > ema(10)",
  "symbol": "BTC",
  "from_date": "2023-01-01",
  "to_date": "2025-08-01",
  "timeframe": "1d",
  "api_url": "https://97ne32z9yn5u.share.zrok.io",
  "backtest_config": {
    "data": "../data/btc_data.csv",
    "signal_column": "buy_sell_signal",
    "buy_value": 1,
    "sell_value": -1,
    "initial_cash": 100000,
    "commission_pct": 0.001,
    "slippage_pct": 0.0005,
    "position_mode": "all_in"
  },
  "retry_count": 2
}
```
