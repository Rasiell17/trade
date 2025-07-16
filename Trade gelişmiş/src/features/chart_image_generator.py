import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class ChartImageDataset(Dataset):
    """
    Grafik görüntülerini yükleyen ve işleyen dataset sınıfı
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ChartImageGenerator:
    """
    OHLCV verilerinden grafik görüntüleri oluşturan sınıf
    """
    def __init__(self, width=224, height=224):
        self.width = width
        self.height = height
        self.scaler = MinMaxScaler()
    def create_candlestick_image(self, ohlcv_data, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (index, row) in enumerate(ohlcv_data.iterrows()):
            open_price = row.get('open', row.get('Open'))
            high_price = row.get('high', row.get('High'))
            low_price = row.get('low', row.get('Low'))
            close_price = row.get('close', row.get('Close'))
            color = 'green' if close_price > open_price else 'red'
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
            body_height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = plt.Rectangle((i-0.4, bottom), 0.8, body_height, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
        ax.set_xlim(-0.5, len(ohlcv_data) - 0.5)
        # Y ekseni için uygun sütunları bul
        low_col = 'low' if 'low' in ohlcv_data.columns else 'Low'
        high_col = 'high' if 'high' in ohlcv_data.columns else 'High'
        ax.set_ylim(ohlcv_data[[low_col, high_col]].min().min() * 0.99,
                    ohlcv_data[[low_col, high_col]].max().max() * 1.01)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img
    def detect_pattern(self, ohlcv_data):
        close_col = 'close' if 'close' in ohlcv_data.columns else 'Close'
        closes = ohlcv_data[close_col].values
        if len(closes) < 10:
            return "undefined"
        recent_closes = closes[-10:]
        trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        volatility = np.std(recent_closes) / np.mean(recent_closes)
        if trend_slope > 0.001 and volatility < 0.02:
            return "bullish_trend"
        elif trend_slope < -0.001 and volatility < 0.02:
            return "bearish_trend"
        elif volatility > 0.05:
            return "consolidation"
        elif abs(trend_slope) < 0.0005:
            return "sideways"
        else:
            return "undefined" 