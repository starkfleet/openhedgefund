# Binance Data API

A FastAPI service for retrieving real-time and historical cryptocurrency data from Binance.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the `binance_data` directory with your Binance API credentials:
```bash
BINANCE_KEY=your_binance_api_key_here
BINANCE_SEC=your_binance_secret_key_here
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

- `GET /` - API information
- `POST /data` - Get historical OHLCV data with parameters
- `GET /data/{symbol}` - Get current data for a specific symbol
- `GET /health` - Health check with Binance API status

## Data Request Format

```json
{
  "name": "BTC|ETH|SOL",
  "from_date": "2024-01-01",
  "to_date": "2024-01-15",
  "timeframe": "1d|1h|15min"
}
```

## Response Format

The API returns OHLCV (Open, High, Low, Close, Volume) data in the same format as before, but now using real Binance data instead of mock data.

## Supported Symbols

- BTC (Bitcoin)
- ETH (Ethereum)  
- SOL (Solana)

## Supported Timeframes

- 1d (1 day)
- 1h (1 hour)
- 15min (15 minutes)

## Notes

- Make sure your Binance API key has read permissions
- The API will automatically convert symbol names to Binance format (e.g., BTC → BTCUSDT)
- All timestamps are returned in UTC
- The health check endpoint will show if the Binance API connection is working properly 