import torch
import numpy as np
import os
import time

from model import LSTMAutoencoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
WINDOW_SIZE = 64

class LSTMAnomalyDetectorRealtime:
    def __init__(self, model_path=None, scaler_mean_path=None, scaler_scale_path=None, threshold=None):
        model_path = model_path or os.path.join(MODELS_DIR, "lstm_autoencoder.pt")
        scaler_mean_path = scaler_mean_path or os.path.join(MODELS_DIR, "scaler_mean.npy")
        scaler_scale_path = scaler_scale_path or os.path.join(MODELS_DIR, "scaler_scale.npy")

        self.model = LSTMAutoencoder()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.mean = np.load(scaler_mean_path)[0]
        self.scale = np.load(scaler_scale_path)[0]
        self.threshold = threshold
        self.buffer = []

    def _normalize(self, window):
        return (window - self.mean) / self.scale

    def push_sample(self, value):
        self.buffer.append(value)
        if len(self.buffer) > WINDOW_SIZE:
            self.buffer.pop(0)

        if len(self.buffer) < WINDOW_SIZE:
            return None

        window = np.array(self.buffer)
        window = self._normalize(window)
        x = torch.tensor(window, dtype=torch.float32).view(1, WINDOW_SIZE, 1)

        start = time.time()
        with torch.no_grad():
            out = self.model(x)
            error = torch.mean((out - x) ** 2).item()
        latency_ms = (time.time() - start) * 1000

        is_anomaly = error > self.threshold if self.threshold else None
        return {"error": error, "is_anomaly": is_anomaly, "latency_ms": latency_ms}

if __name__ == "__main__":
    detector = LSTMAnomalyDetectorRealtime(threshold=0.05)
    simulated_stream = np.random.randn(2000)
    for value in simulated_stream:
        result = detector.push_sample(value)
        if result:
            print(result)
