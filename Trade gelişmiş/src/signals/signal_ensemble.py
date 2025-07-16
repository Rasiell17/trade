from typing import Dict, Any, Optional
import numpy as np

class EnsembleSignalCombiner:
    """
    ML, LSTM, klasik CNN ve görsel CNN (ChartPatternCNN) sinyallerini birleştirir.
    Oylama, ağırlık ve güven skoruna göre nihai sinyal üretir.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Model ağırlıkları: toplamı 1 olmalı
        self.weights = weights or {
            'ml': 0.3,
            'lstm': 0.25,
            'cnn': 0.2,
            'chart_cnn': 0.25
        }
        self.signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        self.reverse_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}

    def combine_signals(self, model_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        model_outputs: {
            'ml': {'signal': 'BUY', 'confidence': 0.8},
            'lstm': {'signal': 'HOLD', 'confidence': 0.6},
            'cnn': {'signal': 'SELL', 'confidence': 0.7},
            'chart_cnn': {'signal': 'BUY', 'confidence': 0.9, 'pattern': 'bullish_trend'}
        }
        """
        votes = []
        weighted_scores = []
        confidences = []
        for model, output in model_outputs.items():
            signal = output['signal']
            conf = output.get('confidence', 1.0)
            weight = self.weights.get(model, 0.0)
            votes.append(self.signal_map.get(signal, 0))
            weighted_scores.append(self.signal_map.get(signal, 0) * weight * conf)
            confidences.append(conf)
        # Oylama
        vote_sum = np.sum(votes)
        # Ağırlıklı skor
        weighted_sum = np.sum(weighted_scores)
        # Nihai karar
        if weighted_sum > 0.2:
            final_signal = 'BUY'
        elif weighted_sum < -0.2:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        # Ek bilgiler
        avg_confidence = float(np.mean(confidences))
        details = {
            'model_signals': {k: v['signal'] for k, v in model_outputs.items()},
            'model_confidences': {k: v.get('confidence', 1.0) for k, v in model_outputs.items()},
            'weighted_sum': float(weighted_sum),
            'vote_sum': int(vote_sum)
        }
        return {
            'final_signal': final_signal,
            'confidence': avg_confidence,
            'details': details
        } 