import pandas as pd
import numpy as np
import torch
from src.models.ensemble_model import EnsemblePredictor

def generate_dummy_df(n=200):
    np.random.seed(42)
    df = pd.DataFrame({
        'open': np.random.rand(n) * 100,
        'high': np.random.rand(n) * 100 + 1,
        'low': np.random.rand(n) * 100 - 1,
        'close': np.random.rand(n) * 100,
        'volume': np.random.rand(n) * 1000,
        'rsi': np.random.rand(n) * 100,
        'macd': np.random.randn(n),
        'macd_signal': np.random.randn(n),
        'bb_percent': np.random.rand(n),
        'bb_width': np.random.rand(n),
        'stoch_k': np.random.rand(n) * 100,
        'stoch_d': np.random.rand(n) * 100,
        'adx': np.random.rand(n) * 50,
        'cci': np.random.randn(n) * 100,
        'williams_r': np.random.rand(n) * -100,
        'mfi': np.random.rand(n) * 100,
        'obv': np.random.randn(n) * 1000,
        'atr': np.random.rand(n) * 10,
        'volume_sma': np.random.rand(n) * 1000,
        'bullish_engulfing': np.random.randint(0, 2, n),
        'bearish_engulfing': np.random.randint(0, 2, n),
        'hammer': np.random.randint(0, 2, n),
        'doji': np.random.randint(0, 2, n),
        'inside_bar': np.random.randint(0, 2, n),
        'outside_bar': np.random.randint(0, 2, n),
        'bos_bullish': np.random.randint(0, 2, n),
        'bos_bearish': np.random.randint(0, 2, n),
    })
    return df

def test_ensemble_predictor():
    config = {}
    predictor = EnsemblePredictor(config)
    df = generate_dummy_df()
    # ML
    X_ml, y_ml = predictor.prepare_features(df)
    predictor.train_ml_model(X_ml, y_ml)
    preds_ml = predictor.predict_ml(X_ml)
    print('ML preds:', np.unique(preds_ml, return_counts=True))
    # LSTM
    X_lstm, y_lstm = predictor.prepare_lstm_data(df)
    predictor.train_lstm_model(X_lstm, y_lstm, epochs=2, batch_size=16)
    preds_lstm = predictor.predict_lstm(X_lstm[:32])
    print('LSTM preds:', np.unique(preds_lstm, return_counts=True))
    # CNN
    X_cnn, y_cnn = predictor.prepare_cnn_data(df)
    predictor.train_cnn_model(X_cnn, y_cnn, epochs=2, batch_size=16)
    preds_cnn = predictor.predict_cnn(X_cnn[:32])
    print('CNN preds:', np.unique(preds_cnn, return_counts=True))

if __name__ == "__main__":
    test_ensemble_predictor() 