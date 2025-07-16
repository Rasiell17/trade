import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from features.chart_image_generator import ChartImageGenerator, ChartImageDataset

class ChartPatternCNN(nn.Module):
    """
    Grafik paternlerini tanımak için özel CNN modeli
    """
    def __init__(self, num_classes=5, input_channels=3, dropout_rate=0.3):
        super(ChartPatternCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.adaptive_pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class CNNTradeAnalyzer:
    """
    CNN tabanlı trade analiz sistemi
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChartPatternCNN(num_classes=5)
        self.model.to(self.device)
        self.pattern_classes = {
            0: "bullish_trend",
            1: "bearish_trend",
            2: "consolidation",
            3: "sideways",
            4: "undefined"
        }
        self.reverse_pattern_classes = {v: k for k, v in self.pattern_classes.items()}
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_generator = ChartImageGenerator()
        if model_path:
            self.load_model(model_path)
    def prepare_training_data(self, ohlcv_data_list, window_size=50):
        images = []
        labels = []
        for ohlcv_data in ohlcv_data_list:
            ohlcv_data = ohlcv_data.copy()
            ohlcv_data.columns = [c.lower() for c in ohlcv_data.columns]
            for i in range(window_size, len(ohlcv_data)):
                window_data = ohlcv_data.iloc[i-window_size:i]
                image = self.image_generator.create_candlestick_image(window_data)
                pattern = self.image_generator.detect_pattern(window_data)
                if pattern in self.reverse_pattern_classes:
                    images.append(image)
                    labels.append(self.reverse_pattern_classes[pattern])
        return np.array(images), np.array(labels)
    def train_model(self, train_images, train_labels, val_images=None, val_labels=None, epochs=50, batch_size=32, learning_rate=0.001):
        train_dataset = ChartImageDataset(train_images, train_labels, self.transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_images is not None and val_labels is not None:
            val_dataset = ChartImageDataset(val_images, val_labels, self.transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            if val_images is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}')
            scheduler.step()
        return train_losses, val_losses
    def predict_pattern(self, ohlcv_data):
        self.model.eval()
        ohlcv_data = ohlcv_data.copy()
        ohlcv_data.columns = [c.lower() for c in ohlcv_data.columns]
        image = self.image_generator.create_candlestick_image(ohlcv_data)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        pattern_name = self.pattern_classes[predicted_class]
        return {
            'pattern': pattern_name,
            'confidence': confidence,
            'probabilities': {self.pattern_classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
    def generate_trade_signal(self, ohlcv_data, confidence_threshold=0.7):
        prediction = self.predict_pattern(ohlcv_data)
        if prediction['confidence'] < confidence_threshold:
            return {
                'signal': 'HOLD',
                'confidence': prediction['confidence'],
                'pattern': prediction['pattern'],
                'reason': 'Low confidence'
            }
        pattern = prediction['pattern']
        confidence = prediction['confidence']
        if pattern == 'bullish_trend':
            signal = 'BUY'
        elif pattern == 'bearish_trend':
            signal = 'SELL'
        elif pattern == 'consolidation':
            signal = 'HOLD'
        elif pattern == 'sideways':
            signal = 'HOLD'
        else:
            signal = 'HOLD'
        return {
            'signal': signal,
            'confidence': confidence,
            'pattern': pattern,
            'reason': f'Pattern: {pattern} with {confidence:.2%} confidence'
        }
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pattern_classes': self.pattern_classes
        }, path)
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pattern_classes = checkpoint['pattern_classes']
        self.reverse_pattern_classes = {v: k for k, v in self.pattern_classes.items()} 