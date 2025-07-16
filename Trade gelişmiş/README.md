# Hibrit Yapay Zeka Destekli Gerçek Zamanlı Trade Analiz ve Sinyal Sistemi

## Proje Amacı
Bu proje; kripto para, hisse senedi ve forex piyasalarında teknik analiz, fiyat aksiyonu, zaman serisi ve görsel formasyonları birleştirerek çalışan, gerçek zamanlı ve akıllı trade sinyalleri üreten modüler bir analiz altyapısı sunar.

## Temel Özellikler
- **Gerçek zamanlı veri çekme (Binance/CCXT)**
- **Teknik gösterge ve price action hesaplama**
- **Makine öğrenmesi (ML), LSTM ve CNN tabanlı sinyal üretimi**
- **Görsel CNN ile grafik formasyon tanıma**
- **Ensemble (oylama, ağırlık, güven skoru) ile nihai sinyal**
- **Modüler, test edilebilir ve genişletilebilir Python altyapısı**

---

## Klasör ve Modül Yapısı

```
src/
├── config/           # Ayarlar ve ortam değişkenleri
├── data/             # Veri çekme, ön işleme ve yönetimi
├── features/         # Görsel ve veri tabanlı feature engineering
├── indicators/       # Teknik indikatör hesaplama
├── models/           # ML, LSTM, CNN, Ensemble ve Görsel CNN modelleri
├── patterns/         # Price action ve formasyon analizleri
├── signals/          # Sinyal birleştirici ve yönetici
├── utils/            # Yardımcı fonksiyonlar
```

---

## Kurulum

1. **Gereksinimler**
   - Python 3.8+
   - Tüm bağımlılıklar için:
     ```
     pip install -r requirements.txt
     ```
   - TA-Lib için Windows .whl dosyası gerekebilir:  
     [TA-Lib .whl indir](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

2. **Binance API Anahtarları**
   - Proje kök dizinine `.env` dosyası oluşturun:
     ```
     BINANCE_API_KEY=xxx
     BINANCE_API_SECRET=yyy
     ```
   - Anahtarlarınız kodda görünmez, otomatik olarak ortamdan çekilir.

---

## Modüller ve Akış

### 1. **Veri Çekme**
```python
from src.data.data_fetcher import fetch_ohlcv
df = fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=500)
```

### 2. **Teknik Göstergeler**
```python
from src.indicators.technical_indicators import TechnicalIndicators
ti = TechnicalIndicators()
df = ti.calculate_all_indicators(df)
```

### 3. **Price Action & Formasyon Analizi**
```python
from src.patterns.price_action import PriceActionAnalyzer
pa = PriceActionAnalyzer()
df = pa.analyze_patterns(df)
```

### 4. **Makine Öğrenmesi & Derin Öğrenme Modelleri**
```python
from src.models.ensemble_model import EnsemblePredictor
predictor = EnsemblePredictor({})
X_ml, y_ml = predictor.prepare_features(df)
predictor.train_ml_model(X_ml, y_ml)
preds_ml = predictor.predict_ml(X_ml[-1:])
```
- LSTM ve klasik CNN için de benzer şekilde `prepare_lstm_data`, `train_lstm_model`, `prepare_cnn_data`, `train_cnn_model` fonksiyonları kullanılır.

### 5. **Görsel CNN ile Grafik Analizi**
```python
from src.models.chart_pattern_cnn import CNNTradeAnalyzer
chart_cnn = CNNTradeAnalyzer()
recent_data = df.tail(50)
chart_cnn_pred = chart_cnn.predict_pattern(recent_data)
```

### 6. **Ensemble Sinyal Birleştirici**
```python
from src.signals.signal_ensemble import EnsembleSignalCombiner
combiner = EnsembleSignalCombiner()
model_outputs = {
    'ml': {'signal': 'BUY', 'confidence': 0.8},
    'lstm': {'signal': 'HOLD', 'confidence': 0.7},
    'cnn': {'signal': 'SELL', 'confidence': 0.6},
    'chart_cnn': {'signal': 'BUY', 'confidence': 0.9, 'pattern': 'bullish_trend'}
}
ensemble_result = combiner.combine_signals(model_outputs)
print(ensemble_result)
```

---

## Uçtan Uca Pipeline Örneği

`tests/test_real_pipeline.py` dosyasında gerçek veriyle uçtan uca örnek akış:
```python
# 1. Veri çek
df = fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=500)
# 2. Teknik göstergeler
df = ti.calculate_all_indicators(df)
# 3. Price action
df = pa.analyze_patterns(df)
# 4. ML, LSTM, CNN, Görsel CNN ile sinyal üret
# 5. Ensemble ile nihai sinyal
```
Tüm detaylı örnek için dosyayı inceleyin.

---

## Kullanılan Kütüphaneler

- ccxt, pandas, numpy, scikit-learn, torch, matplotlib, streamlit, pymongo, ta-lib, python-telegram-bot, discord.py

---

## Notlar ve Geliştirici İpuçları

- Her modül bağımsız test edilebilir ve genişletilebilir.
- Ortam değişkenleri `.env` dosyasından otomatik çekilir.
- Kod altyapısı kurumsal, modüler ve production’a uygun şekilde tasarlanmıştır.
- Gelişmiş backtest, risk yönetimi ve canlı sinyal yayını için ek modüller kolayca entegre edilebilir.

---

Her türlü katkı, öneri ve hata bildirimi için iletişime geçebilirsiniz. 