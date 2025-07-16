import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import numpy as np
import torch
from data.data_fetcher import fetch_ohlcv
from indicators.technical_indicators import TechnicalIndicators
from patterns.price_action import PriceActionAnalyzer
from models.ensemble_model import EnsemblePredictor
from models.chart_pattern_cnn import CNNTradeAnalyzer
from signals.signal_ensemble import EnsembleSignalCombiner

# 1. Gerçek OHLCV verisi çek (örnek: BTC/USDT, 1h, 500 mum)
df = fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=500)

# 2. Teknik göstergeleri ekle
ti = TechnicalIndicators()
df = ti.calculate_all_indicators(df)

# 3. Price action patternlerini ekle
pa = PriceActionAnalyzer()
df = pa.analyze_patterns(df)

# 4. Küçük harfli alias sütunlar ekle (gerekirse)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns:
        df[col.lower()] = df[col]

# 5. ML, LSTM, klasik CNN modelleri
predictor = EnsemblePredictor({})
X_ml, y_ml = predictor.prepare_features(df)
predictor.train_ml_model(X_ml, y_ml)
preds_ml = predictor.predict_ml(X_ml[-1:])
ml_signal = { 'signal': predictor.label_mapping.get(preds_ml[0], 'HOLD'), 'confidence': 0.8 }

X_lstm, y_lstm = predictor.prepare_lstm_data(df)
predictor.train_lstm_model(X_lstm, y_lstm, epochs=2, batch_size=16)
preds_lstm = predictor.predict_lstm(X_lstm[-1:])
lstm_signal = { 'signal': predictor.label_mapping.get(preds_lstm[0], 'HOLD'), 'confidence': 0.7 }

X_cnn, y_cnn = predictor.prepare_cnn_data(df)
predictor.train_cnn_model(X_cnn, y_cnn, epochs=2, batch_size=16)
preds_cnn = predictor.predict_cnn(X_cnn[-1:])
cnn_signal = { 'signal': predictor.label_mapping.get(preds_cnn[0], 'HOLD'), 'confidence': 0.6 }

# 6. Görsel CNN (ChartPatternCNN)
chart_cnn = CNNTradeAnalyzer()
recent_data = df.tail(50)
chart_cnn_pred = chart_cnn.predict_pattern(recent_data)
chart_cnn_signal = {
    'signal': 'BUY' if chart_cnn_pred['pattern'] == 'bullish_trend' else (
        'SELL' if chart_cnn_pred['pattern'] == 'bearish_trend' else 'HOLD'),
    'confidence': chart_cnn_pred['confidence'],
    'pattern': chart_cnn_pred['pattern']
}

# 7. Ensemble sinyal birleştirici
combiner = EnsembleSignalCombiner()
model_outputs = {
    'ml': ml_signal,
    'lstm': lstm_signal,
    'cnn': cnn_signal,
    'chart_cnn': chart_cnn_signal
}
ensemble_result = combiner.combine_signals(model_outputs)

print("=== Model Sinyalleri ===")
for k, v in model_outputs.items():
    print(f"{k}: {v}")
print("\n=== Ensemble Nihai Sinyal ===")
print(ensemble_result) 