import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

class PatternType(Enum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    BOS = "break_of_structure"
    CHOCH = "change_of_character"

class PriceActionAnalyzer:
    """Price Action pattern analiz sınıfı"""
    def __init__(self):
        self.patterns = {}

    def analyze_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df = self._detect_candlestick_patterns(result_df)
        result_df = self._detect_bar_patterns(result_df)
        result_df = self._detect_structure_patterns(result_df)
        return result_df

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['bullish_engulfing'] = self._is_bullish_engulfing(df)
        df['bearish_engulfing'] = self._is_bearish_engulfing(df)
        df['hammer'] = self._is_hammer(df)
        df['doji'] = self._is_doji(df)
        return df

    def _is_bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        cond1 = df['close'].shift(1) < df['open'].shift(1)
        cond2 = df['close'] > df['open']
        cond3 = df['open'] < df['close'].shift(1)
        cond4 = df['close'] > df['open'].shift(1)
        return cond1 & cond2 & cond3 & cond4

    def _is_bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        cond1 = df['close'].shift(1) > df['open'].shift(1)
        cond2 = df['close'] < df['open']
        cond3 = df['open'] > df['close'].shift(1)
        cond4 = df['close'] < df['open'].shift(1)
        return cond1 & cond2 & cond3 & cond4

    def _is_hammer(self, df: pd.DataFrame) -> pd.Series:
        cond1 = df['lower_shadow'] > 2 * df['body']
        cond2 = df['upper_shadow'] < df['body'] * 0.1
        cond3 = df['body'] > 0
        return cond1 & cond2 & cond3

    def _is_doji(self, df: pd.DataFrame) -> pd.Series:
        return df['body'] < df['total_range'] * 0.05

    def _detect_bar_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        return df

    def _detect_structure_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['swing_high'] = df['high'].rolling(window=5*2+1, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=5*2+1, center=True).min() == df['low']
        df['bos_bullish'] = df['close'] > df['high'].where(df['swing_high']).ffill().shift(1)
        df['bos_bearish'] = df['close'] < df['low'].where(df['swing_low']).ffill().shift(1)
        return df 