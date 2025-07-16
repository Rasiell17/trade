import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import talib

class TechnicalIndicators:
    """Teknik gösterge hesaplama sınıfı"""
    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df = self._add_trend_indicators(result_df)
        result_df = self._add_momentum_indicators(result_df)
        result_df = self._add_volatility_indicators(result_df)
        result_df = self._add_volume_indicators(result_df)
        result_df = self._add_support_resistance(result_df)
        result_df = self._add_fibonacci_levels(result_df)
        return result_df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_middle'] = dc.donchian_channel_mband()
        df['dc_lower'] = dc.donchian_channel_lband()
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['ad'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        return df

    def _add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        df['swing_high'] = df['high'].rolling(window=window).max()
        df['swing_low'] = df['low'].rolling(window=window).min()
        return df

    def _add_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        recent_high = df['high'].rolling(window=lookback).max()
        recent_low = df['low'].rolling(window=lookback).min()
        diff = recent_high - recent_low
        df['fib_0'] = recent_low
        df['fib_236'] = recent_low + 0.236 * diff
        df['fib_382'] = recent_low + 0.382 * diff
        df['fib_500'] = recent_low + 0.500 * diff
        df['fib_618'] = recent_low + 0.618 * diff
        df['fib_786'] = recent_low + 0.786 * diff
        df['fib_1000'] = recent_high
        return df 