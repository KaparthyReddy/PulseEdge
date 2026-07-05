import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_all_signals
from preprocess import prepare_dataset
from model import LSTMAutoencoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EPOCHS = 50
BATCH_SIZE = 64
LR = 5e-3

def train():
    signals = load_all_signals()
    normal_windows, _, scaler = prepare_dataset(signals)

    X = torch.tensor(normal_windows, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.6f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_autoencoder.pt"))
    np.save(os.path.join(MODELS_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODELS_DIR, "scaler_scale.npy"), scaler.scale_)
    print("Model saved")

if __name__ == "__main__":
    train()
