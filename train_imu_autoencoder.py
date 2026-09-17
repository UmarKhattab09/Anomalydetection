"""
IMU Anomaly Detection — LSTM Autoencoder Trainer (per-device, IMU-only)
========================================================================
Trains one model per device_id, unsupervised, on IMU channels only (robot joint
data is validation-only ground truth and is NEVER fed to the model). Splits are
done at the SESSION level (see pipeline_common.session_level_split) to avoid the
window-level leakage that would come from splitting a sliding window sequence.

Usage
-----
  python train_imu_autoencoder.py --output_dir ./out/models
  python train_imu_autoencoder.py --output_dir ./out/models --device_id 279
  python train_imu_autoencoder.py --output_dir ./out/models --include_devices 280

Requirements: pip install torch pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import LSTMAutoencoder
from pipeline_common import (
    EXCLUDED_DEVICES_DEFAULT,
    WINDOW_SIZE,
    WINDOW_STEP,
    DeviceSplit,
    ZScoreNormalizer,
    load_all_device_sessions,
    make_windows,
    print_sampling_rate_report,
    session_level_split,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─── Config ─────────────────────────────────────────────────────────────────

# LSTM autoencoder architecture — encoder LSTM(64)->LSTM(32), RepeatVector,
# decoder LSTM(32)->LSTM(64)->TimeDistributed(Dense). See models/autoencoder.py.
ENC_HIDDEN1 = 64
ENC_HIDDEN2 = 32
DEC_HIDDEN1 = 32
DEC_HIDDEN2 = 64

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15

# Anomaly threshold: percentile of reconstruction error on VALIDATION windows
# (held-out sessions never trained on), set per device.
THRESHOLD_PERCENTILE = 95


# ─── Per-device training ───────────────────────────────────────────────────

def build_windows_for_sessions(sessions, channels: list[str], normalizer: ZScoreNormalizer,
                                window_size: int, window_step: int) -> np.ndarray:
    all_windows = []
    for s in sessions:
        arr = normalizer.transform(s.df[channels].values.astype(np.float64))
        windows, _ = make_windows(arr, window_size, window_step)
        if len(windows):
            all_windows.append(windows)
    if not all_windows:
        return np.empty((0, window_size, len(channels)), dtype=np.float32)
    return np.concatenate(all_windows, axis=0)


def train_one_device(device_id: int, split: DeviceSplit, output_dir: str,
                      cfg: dict, torch_device: torch.device) -> dict | None:
    print(f"\n{'=' * 60}\n Training model for Device {device_id}\n{'=' * 60}")

    if not split.train:
        print(f"  [skip] no training sessions for device {device_id}")
        return None

    channels = split.train[0].channels
    for s in split.train + split.val + split.test:
        channels = [c for c in channels if c in s.channels]
    if not channels:
        print(f"  [skip] device {device_id}: no channel set common to all its sessions")
        return None

    normalizer = ZScoreNormalizer(channels=channels).fit(split.train)

    train_windows = build_windows_for_sessions(split.train, channels, normalizer,
                                                cfg["window_size"], cfg["window_step"])
    val_windows = build_windows_for_sessions(split.val, channels, normalizer,
                                              cfg["window_size"], cfg["window_step"]) if split.val else train_windows[-max(1, len(train_windows) // 10):]

    if len(train_windows) == 0:
        print(f"  [skip] device {device_id}: 0 training windows (sessions too short for window_size={cfg['window_size']})")
        return None

    print(f"  Channels ({len(channels)}): {channels}")
    print(f"  Train sessions: {[s.session_id for s in split.train]} -> {len(train_windows)} windows")
    print(f"  Val sessions:   {[s.session_id for s in split.val]} -> {len(val_windows)} windows")
    print(f"  Test sessions (held out, not used here): {[s.session_id for s in split.test]}")

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)

    train_batch_size = min(cfg["batch_size"], len(train_tensor))
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=train_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=cfg["batch_size"], shuffle=False)

    model = LSTMAutoencoder(
        n_features=len(channels), seq_len=cfg["window_size"],
        enc_hidden1=cfg["enc_hidden1"], enc_hidden2=cfg["enc_hidden2"],
        dec_hidden1=cfg["dec_hidden1"], dec_hidden2=cfg["dec_hidden2"],
    ).to(torch_device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(torch_device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item()
        avg_val = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | train={avg_train:.6f} | val={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch} (no val improvement for {cfg['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f}")

    # ── Threshold from VALIDATION reconstruction error distribution ──
    model.eval()
    with torch.no_grad():
        val_errors = model.reconstruction_error(val_tensor.to(torch_device)).cpu().numpy()
    threshold = float(np.percentile(val_errors, cfg["threshold_percentile"])) if len(val_errors) else float("nan")
    print(f"  Val reconstruction error: mean={val_errors.mean():.6f} std={val_errors.std():.6f}")
    print(f"  Anomaly threshold ({cfg['threshold_percentile']}th pct of VAL error): {threshold:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_device{device_id}.pt")
    meta_path = os.path.join(output_dir, f"model_device{device_id}_meta.json")
    normalizer_path = os.path.join(output_dir, f"normalizer_device{device_id}.json")

    torch.save({
        "model_state": best_state,
        "config": {
            "n_features": len(channels), "seq_len": cfg["window_size"],
            "enc_hidden1": cfg["enc_hidden1"], "enc_hidden2": cfg["enc_hidden2"],
            "dec_hidden1": cfg["dec_hidden1"], "dec_hidden2": cfg["dec_hidden2"],
        },
    }, model_path)
    normalizer.save(normalizer_path)

    meta = {
        "device_id": device_id,
        "channels": channels,
        "window_size": cfg["window_size"],
        "window_step": cfg["window_step"],
        "threshold": threshold,
        "threshold_percentile": cfg["threshold_percentile"],
        "val_error_mean": float(val_errors.mean()) if len(val_errors) else None,
        "val_error_std": float(val_errors.std()) if len(val_errors) else None,
        "best_val_loss": best_val_loss,
        "n_train_windows": len(train_tensor),
        "n_val_windows": len(val_tensor),
        "train_sessions": [s.session_id for s in split.train],
        "val_sessions": [s.session_id for s in split.val],
        "test_sessions": [s.session_id for s in split.test],
        "normalizer_path": normalizer_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {os.path.basename(model_path)} | {os.path.basename(meta_path)}")

    plot_path = os.path.join(output_dir, f"training_curve_device{device_id}.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Device {device_id} — LSTM Autoencoder Training", fontsize=12)
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.hist(val_errors, bins=40, alpha=0.8)
    ax2.axvline(threshold, color="red", lw=2, linestyle="--",
                label=f"Threshold ({cfg['threshold_percentile']}th pct) = {threshold:.5f}")
    ax2.set_xlabel("Reconstruction error (MSE)"); ax2.set_ylabel("Count")
    ax2.set_title("Val error distribution"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {os.path.basename(plot_path)}")

    return meta


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train per-device IMU LSTM Autoencoder")
    parser.add_argument("--output_dir", default="./out/models")
    parser.add_argument("--device_id", type=int, default=None, help="Train only this device (default: all)")
    parser.add_argument("--include_devices", type=int, nargs="+", default=None,
                        help="Override EXCLUDED_DEVICES_DEFAULT to force-include a device (e.g. 280)")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window_step", type=int, default=WINDOW_STEP)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold_pct", type=float, default=THRESHOLD_PERCENTILE)
    args = parser.parse_args()

    excluded = EXCLUDED_DEVICES_DEFAULT.copy()
    if args.include_devices:
        excluded -= set(args.include_devices)

    print("Discovering sessions and verifying sampling rate...")
    device_sessions = load_all_device_sessions(excluded_devices=excluded)
    rate_df = print_sampling_rate_report(device_sessions)

    splits = session_level_split(device_sessions)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {torch_device}")

    cfg = {
        "window_size": args.window, "window_step": args.window_step,
        "enc_hidden1": ENC_HIDDEN1, "enc_hidden2": ENC_HIDDEN2,
        "dec_hidden1": DEC_HIDDEN1, "dec_hidden2": DEC_HIDDEN2,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "threshold_percentile": args.threshold_pct,
    }

    device_ids = [args.device_id] if args.device_id is not None else sorted(splits.keys())

    summary_rows = []
    for did in device_ids:
        if did not in splits:
            print(f"  [skip] device {did}: not found")
            continue
        meta = train_one_device(did, splits[did], args.output_dir, cfg, torch_device)
        if meta is None:
            continue
        device_rates = rate_df[rate_df["device_id"] == did]
        summary_rows.append({
            "device_id": did,
            "median_effective_hz": round(device_rates["effective_hz"].median(), 2) if len(device_rates) else None,
            "n_flagged_sessions": int(device_rates["flagged"].sum()) if len(device_rates) else None,
            "n_train_sessions": len(meta["train_sessions"]),
            "n_val_sessions": len(meta["val_sessions"]),
            "n_test_sessions": len(meta["test_sessions"]),
            "threshold": round(meta["threshold"], 6),
        })

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'=' * 60}\n Per-device training summary\n{'=' * 60}")
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(args.output_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(f"All models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
