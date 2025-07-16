import pandas as pd

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(method='ffill').fillna(method='bfill')

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min()) 