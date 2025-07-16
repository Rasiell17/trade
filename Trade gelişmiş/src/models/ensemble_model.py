import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    """LSTM modeli zaman serisi tahminleri için"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.softmax(out)
        return out

class CNNModel(nn.Module):
    """CNN modeli grafik pattern tanıma için"""
    def __init__(self, input_channels: int = 1, num_classes: int = 3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 64x64 input için
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class EnsemblePredictor:
    """Ensemble model sınıfı - ML, LSTM ve CNN'i birleştiren"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.weights = {'ml': 0.4, 'lstm': 0.35, 'cnn': 0.25}
        self.label_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_percent', 'bb_width',
            'stoch_k', 'stoch_d', 'adx', 'cci', 'williams_r',
            'mfi', 'obv', 'atr', 'volume_sma'
        ]
        pattern_columns = [
            'bullish_engulfing', 'bearish_engulfing', 'hammer', 'doji',
            'inside_bar', 'outside_bar', 'bos_bullish', 'bos_bearish'
        ]
        all_features = feature_columns + pattern_columns
        available_features = [col for col in all_features if col in df.columns]
        X = df[available_features].astype(float).values
        y = self._create_target(df).astype(float)
        return X, y
    def _create_target(self, df: pd.DataFrame, forward_periods: int = 5) -> np.ndarray:
        future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        buy_threshold = 0.02
        sell_threshold = -0.02
        y = np.where(future_returns > buy_threshold, 2, np.where(future_returns < sell_threshold, 0, 1))
        return y
    def prepare_lstm_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_percent']
        available_features = [col for col in feature_columns if col in df.columns]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[available_features].values)
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(self._create_target(df.iloc[i:i+1])[0])
        return torch.FloatTensor(X), torch.LongTensor(y)
    def prepare_cnn_data(self, df: pd.DataFrame, image_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        labels = []
        for i in range(len(df)):
            image = np.random.rand(image_size, image_size)
            images.append(image)
            if i < len(df) - 5:
                label = self._create_target(df.iloc[i:i+1])[0]
                labels.append(label)
        images = np.array(images[:-5])
        labels = np.array(labels)
        return torch.FloatTensor(images).unsqueeze(1), torch.LongTensor(labels)
    def train_ml_model(self, X: np.ndarray, y: np.ndarray):
        from collections import Counter
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        class_counts = Counter(y_clean)
        if min(class_counts.values()) < 2 or len(np.unique(y_clean)) < 2:
            # Stratify olmadan split
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
            )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['ml'] = scaler
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_model)],
            voting='soft',
            weights=[2, 3]
        )
        ensemble.fit(X_train_scaled, y_train)
        self.models['ml'] = ensemble
        self.is_trained = True
        y_pred = ensemble.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"ML Ensemble Test Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        joblib.dump(ensemble, 'ml_ensemble_model.pkl')
        joblib.dump(scaler, 'ml_ensemble_scaler.pkl')
    def predict_ml(self, X: np.ndarray) -> np.ndarray:
        model = self.models.get('ml')
        scaler = self.scalers.get('ml')
        if model is None or scaler is None:
            raise ValueError("ML modeli veya scaler yok!")
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        return preds
    def train_lstm_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        input_size = X.shape[2]
        num_classes = max(2, len(torch.unique(y)))
        model = LSTMModel(input_size=input_size, output_size=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[LSTM] Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        self.models['lstm'] = model
        self.is_trained = True
        torch.save(model.state_dict(), 'lstm_model.pth')
    def predict_lstm(self, X: torch.Tensor) -> np.ndarray:
        model = self.models.get('lstm')
        if model is None:
            raise ValueError("LSTM modeli eğitilmemiş!")
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds
    def train_cnn_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        input_channels = X.shape[1]
        num_classes = max(2, len(torch.unique(y)))
        model = CNNModel(input_channels=input_channels, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[CNN] Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        self.models['cnn'] = model
        self.is_trained = True
        torch.save(model.state_dict(), 'cnn_model.pth')
    def predict_cnn(self, X: torch.Tensor) -> np.ndarray:
        model = self.models.get('cnn')
        if model is None:
            raise ValueError("CNN modeli eğitilmemiş!")
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds 