import pandas as pd

def validate_ohlcv(df: pd.DataFrame) -> bool:
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return False
    return True 