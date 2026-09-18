"""
Autoencoder models + inference helpers.

Single source of truth for the model architectures and scoring logic — imported by
train_imu_autoencoder.py and evaluate.py so training and evaluation never drift apart.

Three architectures, each producing a reconstruction of its input window
(batch, seq_len, n_features):

  dense_ae  feed-forward autoencoder on the flattened window
  lstm_ae   encoder LSTM(64)->LSTM(32), RepeatVector, decoder LSTM(32)->LSTM(64)->Dense
  rnn_ae    same funnel shape as lstm_ae but with plain (tanh) nn.RNN layers

Each model has its own hyperparameters (DEFAULT_HYPERPARAMS, overridden by the tuned
models/best_hyperparams.json when it exists): architecture size, dropout, plus
optimizer / batch / patience / augmentation settings. The LSTM and RNN autoencoders use separate
recurrent modules for the encoder/decoder halves so a funnel shape (hidden size
changing between layers) is possible, unlike a single stacked nn.LSTM/nn.RNN.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _ErrorMixin:
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction. Shape: (batch,)"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))

    def reconstruction_error_per_window(self, x: torch.Tensor) -> np.ndarray:
        return self.reconstruction_error(x).cpu().numpy()

    def reconstruct(self, x: torch.Tensor) -> np.ndarray:
        """Reconstructed windows as a numpy array, same shape as x."""
        with torch.no_grad():
            return self.forward(x).cpu().numpy()


# --------------------------------------------------------------------------
# Dense autoencoder
# --------------------------------------------------------------------------

class DenseAutoencoder(_ErrorMixin, nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden: int = 128, latent: int = 32, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        flat = n_features * seq_len
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, flat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)).view(-1, self.seq_len, self.n_features)


# --------------------------------------------------------------------------
# Recurrent autoencoders (LSTM / plain RNN)
# --------------------------------------------------------------------------

class _RecurrentEncoder(nn.Module):
    def __init__(self, rnn_cls, n_features: int, hidden1: int, hidden2: int, dropout: float = 0.0):
        super().__init__()
        self.rnn1 = rnn_cls(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.rnn2 = rnn_cls(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn1(x)
        _, h_n = self.rnn2(self.drop(out))
        if isinstance(h_n, tuple):  # LSTM returns (h_n, c_n)
            h_n = h_n[0]
        return h_n[-1]  # (batch, hidden2) — the latent/bottleneck vector


class _RecurrentDecoder(nn.Module):
    def __init__(self, rnn_cls, n_features: int, seq_len: int, hidden1: int, hidden2: int, latent: int,
                 dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.drop = nn.Dropout(dropout)
        self.rnn1 = rnn_cls(input_size=latent, hidden_size=hidden1, batch_first=True)
        self.rnn2 = rnn_cls(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dense = nn.Linear(hidden2, n_features)  # applied per-timestep = TimeDistributed(Dense)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # RepeatVector
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(self.drop(out))
        return self.dense(out)


class _RecurrentAutoencoder(_ErrorMixin, nn.Module):
    rnn_cls: type = nn.LSTM

    def __init__(self, n_features: int, seq_len: int,
                 enc_hidden1: int = 64, enc_hidden2: int = 32,
                 dec_hidden1: int = 32, dec_hidden2: int = 64, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.encoder = _RecurrentEncoder(self.rnn_cls, n_features, enc_hidden1, enc_hidden2, dropout)
        self.decoder = _RecurrentDecoder(self.rnn_cls, n_features, seq_len, dec_hidden1, dec_hidden2, enc_hidden2, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class LSTMAutoencoder(_RecurrentAutoencoder):
    rnn_cls = nn.LSTM


class RNNAutoencoder(_RecurrentAutoencoder):
    rnn_cls = nn.RNN


# --------------------------------------------------------------------------
# Registry + per-model hyperparameters
# --------------------------------------------------------------------------

MODEL_CLASSES: dict[str, type] = {
    "dense_ae": DenseAutoencoder,
    "lstm_ae": LSTMAutoencoder,
    "rnn_ae": RNNAutoencoder,
}

# "arch" keys go to the model constructor; the rest are training settings:
#   noise_std   std of Gaussian noise added to training inputs (denoising AE; targets stay clean)
#   train_step  window stride for TRAINING windows (smaller = more, more-overlapping windows).
#               Validation/test windows always use pipeline_common.WINDOW_STEP so thresholds and
#               evaluation labels stay comparable.
DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "dense_ae": {
        "arch": {"hidden": 128, "latent": 32, "dropout": 0.0},
        "epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-5, "patience": 15,
        "noise_std": 0.0, "train_step": 15,
    },
    "lstm_ae": {
        "arch": {"enc_hidden1": 64, "enc_hidden2": 32, "dec_hidden1": 32, "dec_hidden2": 64, "dropout": 0.0},
        "epochs": 100, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-5, "patience": 15,
        "noise_std": 0.0, "train_step": 15,
    },
    "rnn_ae": {
        "arch": {"enc_hidden1": 48, "enc_hidden2": 16, "dec_hidden1": 16, "dec_hidden2": 48, "dropout": 0.0},
        "epochs": 100, "batch_size": 32, "lr": 5e-4, "weight_decay": 1e-4, "patience": 15,
        "noise_std": 0.0, "train_step": 15,
    },
}

# Written by tune_hyperparameters.py: the winning config per model. When present it
# replaces the defaults above, so training and evaluation use the tuned settings.
BEST_HYPERPARAMS_PATH = Path(__file__).parent / "best_hyperparams.json"


def get_hyperparams(model_name: str, use_best: bool = True) -> dict:
    hp = copy.deepcopy(DEFAULT_HYPERPARAMS[model_name])
    if use_best and BEST_HYPERPARAMS_PATH.exists():
        with open(BEST_HYPERPARAMS_PATH) as f:
            best = json.load(f).get(model_name)
        if best:
            hp.update({k: v for k, v in best.items() if k != "arch"})
            hp["arch"] = {**hp["arch"], **best.get("arch", {})}
    return hp


def build_model(model_name: str, n_features: int, seq_len: int, arch: dict | None = None) -> nn.Module:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(MODEL_CLASSES)}")
    arch = DEFAULT_HYPERPARAMS[model_name]["arch"] if arch is None else arch
    return MODEL_CLASSES[model_name](n_features=n_features, seq_len=seq_len, **arch)


def load_model_for_device(device_id: int, model_dir: Path, torch_device: torch.device):
    """
    Load the trained model + calibrated threshold + normalizer for one device.
    model_dir is the folder holding that model's artifacts (e.g. models/trained/lstm_ae).
    Returns (model, meta) or None if no model exists for this device yet.
    meta["normalizer_path"] points at a pipeline_common.ZScoreNormalizer JSON.
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
    model = build_model(ckpt["model_name"], cfg["n_features"], cfg["seq_len"], ckpt["arch"]).to(torch_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, meta
