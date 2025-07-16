import numpy as np
import pandas as pd
import torch
from src.models.ensemble_model import EnsemblePredictor
from src.models.chart_pattern_cnn import CNNTradeAnalyzer
from src.features.chart_image_generator import ChartImageGenerator
from src.signals.signal_ensemble import EnsembleSignalCombiner

# 1. Dummy OHLCV veri oluştur
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='1H')
base_price = 100
ohlcv_data = []
for i in range(len(dates)):
    change = np.random.normal(0, 0.02)
    base_price = base_price * (1 + change)
    high = base_price * (1 + abs(np.random.normal(0, 0.01)))
    low = base_price * (1 - abs(np.random.normal(0, 0.01)))
    open_price = base_price * (1 + np.random.normal(0, 0.005))
    close_price = base_price * (1 + np.random.normal(0, 0.005))
    volume = np.random.randint(1000, 10000)
    ohlcv_data.append({
        'Date': dates[i],
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close_price,
        'Volume': volume
    })
df = pd.DataFrame(ohlcv_data)
# Küçük harfli alias sütunlar ekle
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col.lower()] = df[col]

# 2. Teknik indikatör ve price action dummy feature'ları ekle (gerçek pipeline'da hesaplanır)
df['rsi'] = np.random.rand(len(df)) * 100
df['macd'] = np.random.randn(len(df))
df['macd_signal'] = np.random.randn(len(df))
df['bb_percent'] = np.random.rand(len(df))
df['bb_width'] = np.random.rand(len(df))
df['stoch_k'] = np.random.rand(len(df)) * 100
df['stoch_d'] = np.random.rand(len(df)) * 100
df['adx'] = np.random.rand(len(df)) * 50
df['cci'] = np.random.randn(len(df)) * 100
df['williams_r'] = np.random.rand(len(df)) * -100
df['mfi'] = np.random.rand(len(df)) * 100
df['obv'] = np.random.randn(len(df)) * 1000
df['atr'] = np.random.rand(len(df)) * 10
df['volume_sma'] = np.random.rand(len(df)) * 1000
df['bullish_engulfing'] = np.random.randint(0, 2, len(df))
df['bearish_engulfing'] = np.random.randint(0, 2, len(df))
df['hammer'] = np.random.randint(0, 2, len(df))
df['doji'] = np.random.randint(0, 2, len(df))
df['inside_bar'] = np.random.randint(0, 2, len(df))
df['outside_bar'] = np.random.randint(0, 2, len(df))
df['bos_bullish'] = np.random.randint(0, 2, len(df))
df['bos_bearish'] = np.random.randint(0, 2, len(df))

# 3. ML, LSTM, klasik CNN modelleri (dummy eğitim, gerçek projede model dosyası yüklenir)
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

# 4. Görsel CNN (ChartPatternCNN)
chart_cnn = CNNTradeAnalyzer()
recent_data = df.tail(50)
chart_cnn_pred = chart_cnn.predict_pattern(recent_data)
chart_cnn_signal = {
    'signal': 'BUY' if chart_cnn_pred['pattern'] == 'bullish_trend' else (
        'SELL' if chart_cnn_pred['pattern'] == 'bearish_trend' else 'HOLD'),
    'confidence': chart_cnn_pred['confidence'],
    'pattern': chart_cnn_pred['pattern']
}

# 5. Ensemble sinyal birleştirici
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