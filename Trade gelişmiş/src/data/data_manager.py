import pandas as pd
import os

def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path)

def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True) 