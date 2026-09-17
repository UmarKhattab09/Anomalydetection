"""
LSTM autoencoder model + inference helpers.

Single source of truth for the model architecture and scoring logic — imported by
train_imu_autoencoder.py and evaluate.py so training and evaluation never drift apart.

Architecture (as specified): encoder LSTM(64) -> LSTM(32), RepeatVector, decoder
LSTM(32) -> LSTM(64) -> TimeDistributed(Dense). Unlike a single stacked nn.LSTM
(which forces every layer to the same hidden size), the encoder/decoder halves
here are separate nn.LSTM modules precisely so the 64->32->32->64 funnel shape is
possible — this is a deliberate difference from thesiswork/models/autoencoder.py's
simpler same-size stacked design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(out)
        return h_n[-1]  # (batch, hidden2) — the latent/bottleneck vector


class Decoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden2: int = 32, hidden1: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(input_size=hidden2, hidden_size=hidden2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden2, hidden_size=hidden1, batch_first=True)
        self.dense = nn.Linear(hidden1, n_features)  # applied per-timestep = TimeDistributed(Dense)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # RepeatVector
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return self.dense(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int,
                 enc_hidden1: int = 64, enc_hidden2: int = 32,
                 dec_hidden1: int = 32, dec_hidden2: int = 64):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.encoder = Encoder(n_features, enc_hidden1, enc_hidden2)
        self.decoder = Decoder(n_features, seq_len, dec_hidden1, dec_hidden2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction. Shape: (batch,)"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))

    def reconstruction_error_per_window(self, x: torch.Tensor) -> np.ndarray:
        return self.reconstruction_error(x).cpu().numpy()


def load_model_for_device(device_id: int, model_dir: Path, torch_device: torch.device):
    """
    Load the trained model + calibrated threshold + normalizer for one device.
    Returns (model, meta) or None if no model exists for this device yet.
    meta["normalizer"] is a pipeline_common.ZScoreNormalizer-shaped dict.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / f"model_device{device_id}.pt"
    meta_path = model_dir / f"model_device{device_id}_meta.json"
    if not (model_path.exists() and meta_path.exists()):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    ckpt = torch.load(model_path, map_location=torch_device)
    cfg = ckpt["config"]
    model = LSTMAutoencoder(
        n_features=cfg["n_features"], seq_len=cfg["seq_len"],
        enc_hidden1=cfg["enc_hidden1"], enc_hidden2=cfg["enc_hidden2"],
        dec_hidden1=cfg["dec_hidden1"], dec_hidden2=cfg["dec_hidden2"],
    ).to(torch_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, meta
