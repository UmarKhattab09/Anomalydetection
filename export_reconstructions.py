"""
IMU Anomaly Detection — Reconstructions and evaluation metrics per model
========================================================================
For every trained model (models/trained/<model>/) and device, reconstructs whole sessions
on the VALIDATION and TEST splits and computes reconstruction + detection metrics.

Splits scored
-------------
  val            the device's clean validation sessions (used for early stopping + threshold)
  test           the device's clean held-out test sessions (never used for any decision)
  test_injected  the test sessions with synthetic anomalies injected (out/anomaly_test_sets)
  val_injected   the val sessions with synthetic anomalies (out/anomaly_val_sets). This set drove
                 hyperparameter selection, so treat its numbers as optimistic reference only.

Outputs (per model, under models/trained/<model>/)
--------------------------------------------------
  reconstruction/<split>/device<D>_session<S>.csv   original vs reconstructed signal per sample, in
                                                    original units (NaN where no window covers a sample),
                                                    plus per-sample window error and injected labels
  reconstruction/<split>/device<D>_session<S>.png   original vs reconstruction + window error/threshold
  metrics_reconstruction.csv   per device x split: MSE / MAE / RMSE / R2 (z-scored units) + per-channel MSE,
                               window-error stats and the fraction of windows over the threshold
  metrics_detection.csv        per device x split x score_mode x anomaly type: AUROC, AP, precision/recall/F1
                               (type windows vs. normal windows). score_mode is "two_sided" (default rule: error
                               outside the central 95% of clean val error) or "one_sided" (error above the 95th
                               percentile). See scoring.py for why two-sided.
And one file across models:  models/trained/summary_metrics.csv

Reconstruction metrics are in z-score units (each channel scaled by its train-session std), so
channels are comparable; the CSVs also carry original-unit signals if you want those.
A sample's reconstruction is the mean over all windows covering it; samples in the tail that no
window covers are left blank and excluded from the metrics.

Usage
-----
  python export_reconstructions.py
  python export_reconstructions.py --models lstm_ae --no_plots
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate import label_windows, precision_recall_f1
from generate_anomalies import ANOMALY_TYPES
from models.autoencoder import MODEL_CLASSES, load_model_for_device
from scoring import two_sided_bounds, two_sided_flags, two_sided_score, val_reference_errors
from pipeline_common import (
    EXCLUDED_DEVICES_DEFAULT,
    ZScoreNormalizer,
    load_all_device_sessions,
    make_windows,
    session_level_split,
)

TEST_ANOMALY_DIR = os.path.join("out", "anomaly_test_sets")
VAL_ANOMALY_DIR = os.path.join("out", "anomaly_val_sets")


# --------------------------------------------------------------------------
# Reconstruction of one session
# --------------------------------------------------------------------------

def reconstruct_session(model, normalizer: ZScoreNormalizer, df: pd.DataFrame, channels: list[str],
                        window_size: int, window_step: int, threshold: float,
                        ref_errors: np.ndarray, bounds: tuple[float, float]) -> dict | None:
    arr_z = normalizer.transform(df[channels].values.astype(np.float64))
    windows, starts = make_windows(arr_z, window_size, window_step)
    if not len(windows):
        return None
    x = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        recon_w = model.reconstruct(x)
        win_err = model.reconstruction_error(x).numpy()

    n = len(arr_z)
    recon_sum = np.zeros_like(arr_z)
    count = np.zeros(n)
    err_sum = np.zeros(n)
    for w_i, s in enumerate(starts):
        recon_sum[s:s + window_size] += recon_w[w_i]
        count[s:s + window_size] += 1
        err_sum[s:s + window_size] += win_err[w_i]
    covered = count > 0
    recon_z = np.full_like(arr_z, np.nan)
    recon_z[covered] = recon_sum[covered] / count[covered, None]
    sample_win_err = np.full(n, np.nan)
    sample_win_err[covered] = err_sum[covered] / count[covered]

    return {"arr_z": arr_z, "recon_z": recon_z, "covered": covered, "windows": windows, "starts": starts,
            "win_err": win_err, "sample_win_err": sample_win_err, "threshold": threshold,
            "ref_errors": ref_errors, "bounds": bounds,
            "recon_orig": normalizer.inverse_transform(recon_z)}


def save_session_csv(path: str, df: pd.DataFrame, channels: list[str], rec: dict) -> None:
    out = pd.DataFrame({"sample_idx": np.arange(len(df))})
    if "recorded_at_dt" in df.columns:
        out["recorded_at_dt"] = df["recorded_at_dt"].values
    for j, ch in enumerate(channels):
        out[f"{ch}_orig"] = df[ch].values
        out[f"{ch}_recon"] = rec["recon_orig"][:, j]
    out["window_error"] = rec["sample_win_err"]
    out["flag_one_sided"] = np.where(rec["covered"], rec["sample_win_err"] > rec["threshold"], np.nan)
    out["flag_two_sided"] = np.where(rec["covered"], two_sided_flags(rec["sample_win_err"], rec["bounds"]), np.nan)
    if "anomaly_type" in df.columns:
        out["anomaly_type"] = df["anomaly_type"].values
    out.to_csv(path, index=False)


def plot_session(path: str, title: str, df: pd.DataFrame, channels: list[str], rec: dict,
                 labels: list[dict] | None) -> None:
    n_ch = len(channels)
    cols = 3
    rows = int(np.ceil(n_ch / cols))
    fig = plt.figure(figsize=(16, 2.1 * rows + 3.2))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [1.5])
    x = np.arange(len(df))
    for j, ch in enumerate(channels):
        ax = fig.add_subplot(gs[j // cols, j % cols])
        ax.plot(x, rec["arr_z"][:, j], lw=0.8, color="tab:blue", label="original")
        ax.plot(x, rec["recon_z"][:, j], lw=0.8, color="tab:orange", label="reconstruction")
        for lab in labels or []:
            ax.axvspan(lab["start_idx"], lab["end_idx"], color="red", alpha=0.12)
        ax.set_title(ch, fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    ax = fig.add_subplot(gs[rows, :])
    centers = rec["starts"] + rec["windows"].shape[1] // 2
    ax.plot(centers, rec["win_err"], color="tab:blue", label="window reconstruction error")
    lo, hi = rec["bounds"]
    ax.axhline(hi, color="black", ls="--", label="two-sided upper bound (97.5th pct of val error)")
    ax.axhline(lo, color="dimgray", ls=":", label="two-sided lower bound (2.5th pct of val error)")
    for k, lab in enumerate(labels or []):
        ax.axvspan(lab["start_idx"], lab["end_idx"], color="red", alpha=0.12, label="injected anomaly" if k == 0 else None)
        ax.text((lab["start_idx"] + lab["end_idx"]) / 2, ax.get_ylim()[1] * 0.9, lab["type"], ha="center", fontsize=7)
    ax.set_xlabel("sample index"); ax.set_ylabel("MSE (z-score units)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title + "  (signals shown z-scored)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def reconstruction_metrics(recs: list[dict], channels: list[str]) -> dict:
    """Pooled over all covered samples of the given sessions, z-score units."""
    x = np.concatenate([r["arr_z"][r["covered"]] for r in recs])
    y = np.concatenate([r["recon_z"][r["covered"]] for r in recs])
    err = x - y
    mse_ch = (err ** 2).mean(axis=0)
    var_ch = x.var(axis=0)
    win_err = np.concatenate([r["win_err"] for r in recs])
    thr = recs[0]["threshold"]
    out = {
        "n_samples": len(x), "n_windows": len(win_err),
        "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - mse_ch.sum() / var_ch.sum()) if var_ch.sum() > 0 else float("nan"),
        "window_error_mean": float(win_err.mean()), "window_error_p95": float(np.percentile(win_err, 95)),
        "frac_windows_above_threshold": float((win_err > thr).mean()),
        "frac_windows_flagged_two_sided": float(two_sided_flags(win_err, recs[0]["bounds"]).mean()),
    }
    out.update({f"mse_{ch}": float(m) for ch, m in zip(channels, mse_ch)})
    return out


def detection_metrics(sessions: list[tuple[dict, list[dict]]], mode: str) -> dict[str, dict]:
    """
    Per anomaly type (+ 'all_types'): AUROC/AP threshold-free, P/R/F1 at the val-calibrated rule.
    mode="two_sided": score = distance of the error from the clean-val median (either direction),
                      flag = error outside the central 95% of clean val error.
    mode="one_sided": score = error, flag = error above the 95th percentile of val error.
    """
    parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {t: [] for t in ANOMALY_TYPES + ["all_types"]}
    for rec, labels in sessions:
        gt = label_windows(rec["starts"], rec["windows"].shape[1], labels)
        err = rec["win_err"]
        if mode == "two_sided":
            score, flags = two_sided_score(err, rec["ref_errors"]), two_sided_flags(err, rec["bounds"])
        else:
            score, flags = err, err > rec["threshold"]
        any_pos = np.zeros(len(err), dtype=bool)
        for t in ANOMALY_TYPES:
            any_pos |= gt[t]
            mask = gt[t] | gt["normal"]
            if gt[t].sum() > 0:
                parts[t].append((gt[t][mask], score[mask], flags[mask]))
        mask = any_pos | gt["normal"]
        if any_pos.sum() > 0:
            parts["all_types"].append((any_pos[mask], score[mask], flags[mask]))

    out = {}
    for t, ps in parts.items():
        if not ps:
            continue
        y = np.concatenate([p[0] for p in ps])
        sc = np.concatenate([p[1] for p in ps])
        fl = np.concatenate([p[2] for p in ps])
        if not (0 < y.sum() < len(y)):
            continue
        p, r, f1 = precision_recall_f1(y, fl)
        out[t] = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                  "auroc": float(roc_auc_score(y, sc)), "ap": float(average_precision_score(y, sc)),
                  "precision": p, "recall": r, "f1": f1}
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _load_injected(anomaly_dir: str) -> dict[int, list[dict]]:
    path = os.path.join(anomaly_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        manifest = json.load(f)
    by_device: dict[int, list[dict]] = {}
    for e in manifest:
        by_device.setdefault(e["device_id"], []).append(e)
    return by_device


def export(model_dir: str, model_names: list[str], excluded_devices: set[int], plots: bool) -> None:
    torch.set_num_threads(4)
    device_sessions = load_all_device_sessions(excluded_devices=excluded_devices)
    splits = session_level_split(device_sessions)
    injected = {"test_injected": _load_injected(TEST_ANOMALY_DIR), "val_injected": _load_injected(VAL_ANOMALY_DIR)}

    summary_rows = []
    for name in model_names:
        mdir = os.path.join(model_dir, name)
        recon_rows, det_rows = [], []
        for device_id in sorted(splits):
            loaded = load_model_for_device(device_id, mdir, torch.device("cpu"))
            if loaded is None:
                print(f"  [skip] {name} device {device_id}: not trained")
                continue
            model, meta = loaded
            normalizer = ZScoreNormalizer.load(meta["normalizer_path"])
            channels, ws, wstep, thr = meta["channels"], meta["window_size"], meta["window_step"], meta["threshold"]
            print(f"[{name}] device {device_id}", flush=True)
            ref_errors = val_reference_errors(model, normalizer, splits[device_id].val, channels, ws, wstep)
            bounds = two_sided_bounds(ref_errors)

            # (split name, [(session_id, df, labels-or-None)])
            work: dict[str, list[tuple[str, pd.DataFrame, list[dict] | None]]] = {
                "val": [(s.session_id, s.df, None) for s in splits[device_id].val],
                "test": [(s.session_id, s.df, None) for s in splits[device_id].test],
            }
            for split_name, by_dev in injected.items():
                items = []
                for e in by_dev.get(device_id, []):
                    df = pd.read_csv(e["csv"])
                    df["recorded_at_dt"] = pd.to_datetime(df["recorded_at_dt"])
                    with open(e["labels"]) as f:
                        items.append((e["session_id"], df, json.load(f)))
                work[split_name] = items

            for split_name, items in work.items():
                recs, det_inputs = [], []
                out_dir = os.path.join(mdir, "reconstruction", split_name)
                os.makedirs(out_dir, exist_ok=True)
                for sid, df, labels in items:
                    rec = reconstruct_session(model, normalizer, df, channels, ws, wstep, thr, ref_errors, bounds)
                    if rec is None:
                        continue
                    recs.append(rec)
                    if labels is not None:
                        det_inputs.append((rec, labels))
                    base = os.path.join(out_dir, f"device{device_id}_session{sid}")
                    save_session_csv(base + ".csv", df, channels, rec)
                    if plots:
                        plot_session(base + ".png", f"{name} | device {device_id} | {split_name} | session {sid}",
                                     df, channels, rec, labels)
                if not recs:
                    continue
                if split_name in ("val", "test"):
                    m = reconstruction_metrics(recs, channels)
                    recon_rows.append({"model": name, "device_id": device_id, "split": split_name,
                                       "sessions": ",".join(str(sid) for sid, _, _ in items), "threshold": thr, **m})
                if det_inputs:
                    for mode in ("two_sided", "one_sided"):
                        for atype, dm in detection_metrics(det_inputs, mode).items():
                            det_rows.append({"model": name, "device_id": device_id, "split": split_name,
                                             "score_mode": mode, "anomaly_type": atype, **dm})

        recon_df, det_df = pd.DataFrame(recon_rows), pd.DataFrame(det_rows)
        recon_df.to_csv(os.path.join(mdir, "metrics_reconstruction.csv"), index=False)
        det_df.to_csv(os.path.join(mdir, "metrics_detection.csv"), index=False)

        # Model-level summary: mean across devices (each device has its own threshold/error scale).
        row = {"model": name}
        for split_name in ("val", "test"):
            sub = recon_df[recon_df["split"] == split_name]
            for k in ("mse", "mae", "rmse", "r2", "frac_windows_above_threshold", "frac_windows_flagged_two_sided"):
                row[f"{split_name}_{k}"] = float(sub[k].mean()) if len(sub) else None
        for mode in ("two_sided", "one_sided"):
            for split_name in ("test_injected", "val_injected"):
                base = det_df[(det_df["split"] == split_name) & (det_df["score_mode"] == mode)] if len(det_df) else det_df
                sub = base[base["anomaly_type"] == "all_types"] if len(base) else base
                for k in ("auroc", "ap", "precision", "recall", "f1"):
                    row[f"{mode}_{split_name}_{k}"] = float(sub[k].mean()) if len(sub) else None
                per_type = base[base["anomaly_type"].isin(ANOMALY_TYPES)] if len(base) else base
                row[f"{mode}_{split_name}_mean_type_auroc"] = float(per_type.groupby("anomaly_type")["auroc"].mean().mean()) if len(per_type) else None
                row[f"{mode}_{split_name}_mean_type_f1"] = float(per_type.groupby("anomaly_type")["f1"].mean().mean()) if len(per_type) else None
        summary_rows.append(row)
        print(f"[{name}] wrote metrics + reconstructions under {mdir}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(model_dir, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n" + summary.T.to_string(header=False))
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct val/test sessions and compute metrics per model")
    parser.add_argument("--model_dir", default="./models/trained")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CLASSES), default=None,
                        help="Default: every model with a trained subfolder")
    parser.add_argument("--include_devices", type=int, nargs="+", default=None)
    parser.add_argument("--no_plots", action="store_true", help="Skip PNGs (CSV + metrics only)")
    args = parser.parse_args()

    excluded = EXCLUDED_DEVICES_DEFAULT.copy()
    if args.include_devices:
        excluded -= set(args.include_devices)
    names = args.models or [n for n in sorted(MODEL_CLASSES) if os.path.isdir(os.path.join(args.model_dir, n))]
    export(args.model_dir, names, excluded, plots=not args.no_plots)
