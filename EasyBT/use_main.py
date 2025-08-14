from main import bt
import warnings
warnings.filterwarnings("ignore")

# Example: backtest a pre-labeled CSV like btc_data.csv and generate a report
config = {
    "data": "../btc_data.csv",  # path to CSV
    # optional filters
    # "symbol": "BTCUSDT",
    # "from_date": "2024-01-01",
    # "to_date": "2024-12-31",

    # signal mapping
    "signal_column": "buy_sell_signal",
    "buy_value": 1,
    "sell_value": -1,

    # trading settings
    "initial_cash": 100000,
    "commission_pct": 0.001,  # 0.1% per side
    "slippage_pct": 0.0005,   # 5 bps slippage
    "position_mode": "all_in",  # or: 'fixed_cash', 'fixed_units'
    # "fixed_cash_per_trade": 20000,
    # "fixed_units_per_trade": 0.05,
}

report = bt.backtester_signal_csv_report(config)

print("Summary:\n", report["summary"]) 
print("\nEquity curve (head):\n", report["equity_curve"].head())
print("\nTrades (head):\n", report["trades"].head())