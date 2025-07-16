"""
Sistem genelinde kullanılan sabitler, enum'lar ve metrikler.
"""
from enum import Enum

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    LONG = "LONG"
    SHORT = "SHORT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class TimeFrame(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"

class ModelType(Enum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    CNN = "cnn"
    ENSEMBLE = "ensemble"

class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"

# Teknik göstergeler için sabitler
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'atr_period': 14,
    'volume_sma_period': 20,
    'fibonacci_periods': [23, 38, 50, 61, 78]
}

# Pattern detection thresholds
PATTERN_THRESHOLDS = {
    'doji_threshold': 0.1,
    'hammer_shadow_ratio': 2.0,
    'engulfing_body_ratio': 1.0,
    'breakout_volume_ratio': 1.5
}

# Risk management constants
RISK_CONSTANTS = {
    'kelly_lookback': 252,
    'var_confidence': 0.95,
    'max_correlation': 0.7,
    'volatility_lookback': 30
}

# Backtest constants
BACKTEST_CONSTANTS = {
    'commission_rate': 0.001,
    'slippage_rate': 0.0005,
    'min_trade_amount': 10.0,
    'max_leverage': 3.0
}

# Performance metrics
PERFORMANCE_METRICS = [
    'total_return',
    'sharpe_ratio',
    'calmar_ratio',
    'sortino_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
    'average_trade',
    'total_trades'
] 