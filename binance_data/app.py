from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal
import uvicorn
from datetime import datetime, date
import json
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables
load_dotenv()

# Initialize Binance client
binance_key = os.getenv('BINANCE_KEY')
binance_secret = os.getenv('BINANCE_SEC')

if not binance_key or not binance_secret:
    raise ValueError("BINANCE_KEY and BINANCE_SEC must be set in environment variables")

client = Client(binance_key, binance_secret)

# Initialize FastAPI app
app = FastAPI(
    title="Binance Data API",
    description="A FastAPI service for retrieving binance data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for data structure
class DataRequest(BaseModel):
    name: Literal["BTC", "ETH", "SOL"]
    from_date: date
    to_date: date
    timeframe: Literal["1d", "1h", "15min"]
    
    @field_validator('to_date')
    @classmethod
    def validate_dates(cls, v, info):
        if 'from_date' in info.data and v <= info.data['from_date']:
            raise ValueError('to_date must be after from_date')
        return v

class OHLCVData(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class ApiResponse(BaseModel):
    success: bool
    data: Optional[List[OHLCVData]] = None
    message: Optional[str] = None
    timestamp: datetime

# Helper function to convert Binance timeframe to interval
def get_binance_interval(timeframe: str) -> str:
    timeframe_map = {
        "1d": Client.KLINE_INTERVAL_1DAY,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "15min": Client.KLINE_INTERVAL_15MINUTE
    }
    return timeframe_map.get(timeframe, Client.KLINE_INTERVAL_1DAY)

# Helper function to convert symbol name to Binance format
def get_binance_symbol(name: str) -> str:
    symbol_map = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT", 
        "SOL": "SOLUSDT"
    }
    return symbol_map.get(name, f"{name}USDT")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Binance Data API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "POST /data": "Get binance data with parameters",
            "GET /data/{symbol}": "Get data for specific symbol",
            "GET /health": "Health check"
        }
    }

@app.post("/data", response_model=ApiResponse)
async def get_data(request: DataRequest):
    """Get binance data based on request parameters"""
    try:
        # Convert request parameters to Binance format
        symbol = get_binance_symbol(request.name)
        interval = get_binance_interval(request.timeframe)
        
        # Convert dates to milliseconds timestamp for Binance API
        start_time = int(datetime.combine(request.from_date, datetime.min.time()).timestamp() * 1000)
        end_time = int(datetime.combine(request.to_date, datetime.min.time()).timestamp() * 1000)
        
        # Fetch data from Binance API
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time
        )
        
        # Convert Binance klines to our format
        formatted_data = []
        for kline in klines:
            # Binance klines format: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            formatted_data.append(OHLCVData(
                symbol=symbol,
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
                timestamp=datetime.fromtimestamp(kline[0] / 1000)
            ))
        
        return ApiResponse(
            success=True,
            data=formatted_data,
            message=f"Data for {request.name} from {request.from_date} to {request.to_date} with {request.timeframe} timeframe retrieved successfully",
            timestamp=datetime.utcnow()
        )
    except BinanceAPIException as e:
        raise HTTPException(status_code=400, detail=f"Binance API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@app.get("/data/{symbol}", response_model=ApiResponse)
async def get_data_by_symbol(symbol: str):
    """Get current data for a specific symbol"""
    try:
        # Get 24hr ticker statistics
        ticker = client.get_ticker(symbol=symbol.upper())
        
        # Get recent klines for OHLCV data
        klines = client.get_klines(symbol=symbol.upper(), interval=Client.KLINE_INTERVAL_1HOUR, limit=1)
        
        if not klines:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Use the most recent kline data
        kline = klines[0]
        current_data = OHLCVData(
            symbol=symbol.upper(),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            timestamp=datetime.fromtimestamp(kline[0] / 1000)
        )
        
        return ApiResponse(
            success=True,
            data=[current_data],
            message=f"Data for {symbol} retrieved successfully",
            timestamp=datetime.utcnow()
        )
    except BinanceAPIException as e:
        if e.code == -1121:  # Invalid symbol
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        raise HTTPException(status_code=400, detail=f"Binance API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Binance API connection
        server_time = client.get_server_time()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "binance-data-api",
            "binance_server_time": datetime.fromtimestamp(server_time['serverTime'] / 1000).isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "binance-data-api",
            "error": str(e)
        }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
