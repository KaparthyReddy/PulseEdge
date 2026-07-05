import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])

        seq_len = x.shape[1]
        decoder_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(decoder_input)
        return out
