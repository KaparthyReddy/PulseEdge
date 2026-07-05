import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from data_loader import load_all_signals
from preprocess import prepare_dataset
from model import LSTMAutoencoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def compute_reconstruction_errors(model, X):
    with torch.no_grad():
        out = model(X)
        errors = torch.mean((out - X) ** 2, dim=(1, 2))
    return errors.numpy()

def evaluate():
    signals = load_all_signals()
    normal_windows, anomaly_windows, scaler = prepare_dataset(signals)

    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_autoencoder.pt")))
    model.eval()

    X_normal = torch.tensor(normal_windows, dtype=torch.float32).unsqueeze(-1)
    normal_errors = compute_reconstruction_errors(model, X_normal)
    threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
    print(f"Threshold: {threshold:.6f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label="Normal")

    for key, windows in anomaly_windows.items():
        X_anom = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        errors = compute_reconstruction_errors(model, X_anom)
        detection_rate = np.mean(errors > threshold) * 100
        print(f"{key}: detection rate {detection_rate:.2f}%")
        plt.hist(errors, bins=50, alpha=0.6, label=key)

    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
    print("Plot saved to results/error_distribution.png")

if __name__ == "__main__":
    evaluate()
