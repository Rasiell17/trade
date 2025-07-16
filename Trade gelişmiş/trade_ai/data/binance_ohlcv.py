import ccxt
import pandas as pd
from datetime import datetime

def fetch_ohlcv(symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 500):
    """
    Binance üzerinden OHLCV verisi çeker ve pandas DataFrame olarak döndürür.
    :param symbol: Örn. 'BTC/USDT'
    :param timeframe: '1m', '5m', '15m', '1h', '4h', '1d' vb.
    :param limit: Kaç veri çekilecek (maks. 1000)
    :return: pd.DataFrame
    """
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

if __name__ == "__main__":
    # Örnek kullanım
    df = fetch_ohlcv('BTC/USDT', '1h', 100)
    print(df.tail()) 