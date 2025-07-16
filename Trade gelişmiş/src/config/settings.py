import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Ana konfigürasyon sınıfı"""
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "trading_db")
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/trading")
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "trading_org")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "trading_bucket")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    DISCORD_CHANNEL_ID: str = os.getenv("DISCORD_CHANNEL_ID", "")
    DEFAULT_SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
    DEFAULT_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    MODEL_RETRAIN_INTERVAL: int = 24
    MODEL_VALIDATION_THRESHOLD: float = 0.6
    MAX_POSITION_SIZE: float = 0.1
    STOP_LOSS_THRESHOLD: float = 0.02
    TAKE_PROFIT_THRESHOLD: float = 0.04
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2024-01-01"
    INITIAL_CAPITAL: float = 10000.0
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "trading_system.log")
    CACHE_TTL: int = 300
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "trading_models")
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    class Config:
        env_file = ".env"
        case_sensitive = True
settings = Settings() 