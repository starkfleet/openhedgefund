# openhedgefund
Open source agentic hedge fund

<img width="1530" height="1003" alt="Screenshot 2025-08-14 at 1 09 04 PM" src="https://github.com/user-attachments/assets/86c420b8-7d1d-4ed2-a381-9025f479265f" />

~/Documents/openhedgefund/binance_data python app.py

zrok share public 8000

(.venv)  -> ~/Documents/openhedgefund cd docker_pycode_environment                        
docker run --rm -v "$(pwd)/test_script.py:/test_script.py" py_sandbox python /test_script.py
Executing the sandboxed Python script...
Pandas DataFrame created:
   col1  col2
0     1     3
1     2     4
Numpy array created:
[5 6 7]

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
