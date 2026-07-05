import numpy as np
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 400
STRIDE = 200

def create_windows(signal, window_size=WINDOW_SIZE, stride=STRIDE):
    windows = []
    for start in range(0, len(signal) - window_size, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)

def normalize_windows(windows, scaler=None):
    flat = windows.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        flat = scaler.fit_transform(flat)
    else:
        flat = scaler.transform(flat)
    normalized = flat.reshape(windows.shape)
    return normalized, scaler

def prepare_dataset(signals, normal_key="normal_baseline"):
    normal_windows = create_windows(signals[normal_key])
    normal_windows, scaler = normalize_windows(normal_windows)

    anomaly_windows = {}
    for key, signal in signals.items():
        if key == normal_key:
            continue
        windows = create_windows(signal)
        windows, _ = normalize_windows(windows, scaler)
        anomaly_windows[key] = windows

    return normal_windows, anomaly_windows, scaler
