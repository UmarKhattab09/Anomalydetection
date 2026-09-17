"""
IMU Anomaly Detection — Evaluation (LSTM autoencoder vs. rule-based baselines)
================================================================================
Scores both the trained per-device LSTM autoencoders and the rule-based
baselines (rule_baselines.py) against the SAME held-out, anomaly-injected test
sessions (generate_anomalies.py output), using the injected ground-truth
intervals to compute precision/recall/F1 PER ANOMALY TYPE — not one aggregate
normal-vs-anomaly number.

Usage
-----
  python train_imu_autoencoder.py --output_dir ./out/models      # 1. train
  python generate_anomalies.py --output_dir ./out/anomaly_test_sets  # 2. inject
  python evaluate.py --model_dir ./out/models --anomaly_dir ./out/anomaly_test_sets --output_dir ./out/eval

Outputs (in --output_dir)
--------------------------
  device_summary.csv       per-device: sample rate, session counts, threshold, aggregate test F1
  comparison_table.csv     per anomaly type: LSTM P/R/F1 vs. its rule-based baseline P/R/F1
  overlay_<type>.png       one example per anomaly type: LSTM error + baseline signal +
                            robot-joint ground truth, with the true injected interval shaded
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from generate_anomalies import ANOMALY_TYPES, _reference_channel
from models.autoencoder import load_model_for_device
from pipeline_common import (
    EXCLUDED_DEVICES_DEFAULT,
    ZScoreNormalizer,
    load_all_device_sessions,
    make_windows,
    session_level_split,
)
from rule_baselines import (
    axis_pair_for_channels,
    build_reference_window,
    calibrate_cusum_h,
    cusum_drift_flags,
    dtw_shape_flags,
    jitter_variance_flags,
    rolling_xcorr_flags,
)

OVERLAP_THRESHOLD = 0.5  # fraction of a window that must fall inside a true interval to count as that type


# --------------------------------------------------------------------------
# Ground-truth window labeling
# --------------------------------------------------------------------------

def window_overlap_fraction(w_start: int, w_end: int, i_start: int, i_end: int) -> float:
    inter = max(0, min(w_end, i_end) - max(w_start, i_start))
    return inter / (w_end - w_start)


def label_windows(window_starts: np.ndarray, window_size: int, labels: list[dict]) -> dict[str, np.ndarray]:
    """
    For each anomaly type, returns a boolean array over windows: True where the window
    overlaps that type's true interval by >= OVERLAP_THRESHOLD, aligned to window_starts.
    Also returns "normal": True where a window overlaps NO labeled interval at all.
    """
    n = len(window_starts)
    per_type = {t: np.zeros(n, dtype=bool) for t in ANOMALY_TYPES}
    any_overlap = np.zeros(n, dtype=bool)

    for lab in labels:
        for i, s in enumerate(window_starts):
            e = s + window_size
            frac = window_overlap_fraction(s, e, lab["start_idx"], lab["end_idx"])
            if frac > 0:
                any_overlap[i] = True
            if frac >= OVERLAP_THRESHOLD:
                per_type[lab["type"]][i] = True

    per_type["normal"] = ~any_overlap
    return per_type


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# --------------------------------------------------------------------------
# Per-session scoring
# --------------------------------------------------------------------------

def score_session(csv_path: str, labels_path: str, model, meta: dict, normalizer: ZScoreNormalizer,
                   reference_window: np.ndarray, reference_pool: np.ndarray, torch_device: torch.device,
                   cusum_h: float | None = None) -> dict:
    df = pd.read_csv(csv_path)
    df["recorded_at_dt"] = pd.to_datetime(df["recorded_at_dt"])  # CSV round-trip loses the dtype
    with open(labels_path) as f:
        labels = json.load(f)

    channels = meta["channels"]
    window_size = meta["window_size"]
    window_step = meta["window_step"]
    threshold = meta["threshold"]

    arr_norm = normalizer.transform(df[channels].values.astype(np.float64))
    windows, starts = make_windows(arr_norm, window_size, window_step)
    if len(windows) == 0:
        return {"error": "no windows"}

    with torch.no_grad():
        lstm_errors = model.reconstruction_error(torch.tensor(windows, dtype=torch.float32).to(torch_device)).cpu().numpy()
    lstm_flags = lstm_errors > threshold

    primary_col = _reference_channel(channels)
    primary_idx = channels.index(primary_col)
    primary_series = df[primary_col].values.astype(np.float64)

    drift_flags, cusum_signal = cusum_drift_flags(primary_series, starts, window_size, h=cusum_h)
    jitter_flags, jitter_signal = jitter_variance_flags(primary_series, starts, window_size)
    dtw_flags, dtw_signal = dtw_shape_flags(windows, reference_window, primary_idx, reference_pool=reference_pool)

    axis_pair = axis_pair_for_channels(channels)
    if axis_pair:
        a_col, b_col, needs_diff = axis_pair
        series_a = df[a_col].values.astype(np.float64)
        series_b = df[b_col].values.astype(np.float64)
        if needs_diff:
            series_b = np.gradient(series_b)
        axis_flags, axis_signal = rolling_xcorr_flags(series_a, series_b, starts, window_size)
    else:
        axis_flags, axis_signal = None, None

    gt = label_windows(starts, window_size, labels)

    return {
        "df": df, "labels": labels, "starts": starts, "window_size": window_size,
        "lstm_errors": lstm_errors, "lstm_flags": lstm_flags, "threshold": threshold,
        "drift_flags": drift_flags, "cusum_signal": cusum_signal,
        "jitter_flags": jitter_flags, "jitter_signal": jitter_signal,
        "dtw_flags": dtw_flags, "dtw_signal": dtw_signal,
        "axis_flags": axis_flags, "axis_signal": axis_signal,
        "primary_col": primary_col, "axis_pair": axis_pair,
        "gt": gt,
    }


BASELINE_FOR_TYPE = {
    "drift": "drift_flags",
    "jitter": "jitter_flags",
    "overshoot": "dtw_flags",
    "undershoot": "dtw_flags",
    "stall": "dtw_flags",
    # early_stop wasn't given a dedicated baseline in the spec (only drift/jitter/
    # overshoot+undershoot+stall/axis_decouple were) — it's a shape distortion like
    # overshoot/undershoot/stall (truncated + reversed trajectory), so it uses the
    # same DTW-to-reference baseline rather than going unscored.
    "early_stop": "dtw_flags",
    "axis_decouple": "axis_flags",
}
BASELINE_NAME = {
    "drift_flags": "CUSUM",
    "jitter_flags": "windowed-variance",
    "dtw_flags": "DTW-to-reference",
    "axis_flags": "rolling-xcorr",
}


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def evaluate(model_dir: str, anomaly_dir: str, output_dir: str, excluded_devices: set[int] | None = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(anomaly_dir, "manifest.json")) as f:
        manifest = json.load(f)

    by_device: dict[int, list[dict]] = {}
    for entry in manifest:
        by_device.setdefault(entry["device_id"], []).append(entry)

    # Per-type accumulators: {type: {"lstm": (y_true[], y_pred[]), "baseline": (...)}}
    per_type_true_pred_lstm: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {t: [] for t in ANOMALY_TYPES}
    per_type_true_pred_baseline: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {t: [] for t in ANOMALY_TYPES}

    device_summary_rows = []
    example_for_type: dict[str, dict] = {}

    device_sessions = load_all_device_sessions(excluded_devices=excluded_devices)
    splits = session_level_split(device_sessions)

    for device_id, entries in by_device.items():
        loaded = load_model_for_device(device_id, model_dir, torch_device)
        if loaded is None:
            print(f"  [skip] device {device_id}: no trained model found in {model_dir}")
            continue
        model, meta = loaded
        normalizer = ZScoreNormalizer.load(meta["normalizer_path"])

        # Build the DTW reference + calibration pool from this device's TRAIN windows.
        train_sessions = splits[device_id].train
        train_windows_list = []
        for s in train_sessions:
            arr = normalizer.transform(s.df[meta["channels"]].values.astype(np.float64))
            w, _ = make_windows(arr, meta["window_size"], meta["window_step"])
            if len(w):
                train_windows_list.append(w)
        train_windows = np.concatenate(train_windows_list, axis=0) if train_windows_list else np.empty((0, meta["window_size"], len(meta["channels"])))
        reference_window = build_reference_window(train_windows) if len(train_windows) else None

        primary_col = _reference_channel(meta["channels"])
        train_primary_series = np.concatenate([s.df[primary_col].values.astype(np.float64) for s in train_sessions]) if train_sessions else np.array([])
        cusum_h = calibrate_cusum_h(train_primary_series) if len(train_primary_series) else None

        device_type_f1s = []

        for entry in entries:
            if reference_window is None:
                continue
            result = score_session(entry["csv"], entry["labels"], model, meta, normalizer,
                                    reference_window, train_windows, torch_device, cusum_h=cusum_h)
            if "error" in result:
                continue

            gt = result["gt"]
            normal_mask = gt["normal"]

            for atype in entry["types_injected"]:
                true_mask = gt[atype]
                if true_mask.sum() == 0:
                    continue
                eval_mask = true_mask | normal_mask  # restrict to this type vs. normal (per-type binary task)

                y_true = true_mask[eval_mask]
                y_pred_lstm = result["lstm_flags"][eval_mask]
                per_type_true_pred_lstm[atype].append((y_true, y_pred_lstm))

                baseline_key = BASELINE_FOR_TYPE[atype]
                baseline_flags = result.get(baseline_key)
                if baseline_flags is not None:
                    y_pred_baseline = baseline_flags[eval_mask]
                    per_type_true_pred_baseline[atype].append((y_true, y_pred_baseline))

                _, _, f1 = precision_recall_f1(y_true, y_pred_lstm)
                device_type_f1s.append(f1)

                if atype not in example_for_type:
                    robot_df = next((s.robot_df for s in device_sessions.get(device_id, [])
                                      if s.session_id == entry["session_id"]), None)
                    example_for_type[atype] = {
                        "device_id": device_id, "session_id": entry["session_id"],
                        "result": result, "label": next(l for l in result["labels"] if l["type"] == atype),
                        "robot_df": robot_df,
                    }

        rate_df = pd.DataFrame([{
            "session_id": s.session_id, "effective_hz": s.sample_rate.effective_hz,
        } for s in device_sessions.get(device_id, [])])

        device_summary_rows.append({
            "device_id": device_id,
            "median_effective_hz": round(rate_df["effective_hz"].median(), 2) if len(rate_df) else None,
            "n_train_sessions": len(splits[device_id].train),
            "n_val_sessions": len(splits[device_id].val),
            "n_test_sessions": len(splits[device_id].test),
            "threshold": round(meta["threshold"], 6),
            "mean_test_f1_lstm": round(float(np.mean(device_type_f1s)), 4) if device_type_f1s else None,
        })

    # ── Comparison table: per anomaly type, LSTM vs. its baseline ──
    comparison_rows = []
    for atype in ANOMALY_TYPES:
        lstm_pairs = per_type_true_pred_lstm[atype]
        baseline_pairs = per_type_true_pred_baseline[atype]

        if lstm_pairs:
            y_true = np.concatenate([p[0] for p in lstm_pairs])
            y_pred = np.concatenate([p[1] for p in lstm_pairs])
            lp, lr, lf = precision_recall_f1(y_true, y_pred)
            n_eval = len(y_true)
        else:
            lp = lr = lf = None
            n_eval = 0

        if baseline_pairs:
            y_true_b = np.concatenate([p[0] for p in baseline_pairs])
            y_pred_b = np.concatenate([p[1] for p in baseline_pairs])
            bp, br, bf = precision_recall_f1(y_true_b, y_pred_b)
            baseline_name = BASELINE_NAME[BASELINE_FOR_TYPE[atype]]
        else:
            bp = br = bf = None
            baseline_name = BASELINE_NAME[BASELINE_FOR_TYPE[atype]]

        comparison_rows.append({
            "anomaly_type": atype, "n_eval_windows": n_eval,
            "lstm_precision": lp, "lstm_recall": lr, "lstm_f1": lf,
            "baseline": baseline_name,
            "baseline_precision": bp, "baseline_recall": br, "baseline_f1": bf,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    summary_df = pd.DataFrame(device_summary_rows)

    print(f"\n{'=' * 70}\n Per-device summary\n{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"\n{'=' * 70}\n LSTM vs. rule-based baseline, per anomaly type\n{'=' * 70}")
    print(comparison_df.to_string(index=False))

    summary_path = os.path.join(output_dir, "device_summary.csv")
    comparison_path = os.path.join(output_dir, "comparison_table.csv")
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved: {summary_path}\nSaved: {comparison_path}")

    # ── Overlay plots — one example per anomaly type ──
    for atype, ex in example_for_type.items():
        plot_overlay(atype, ex, output_dir)


def _signal_for_type(atype: str, result: dict) -> tuple[np.ndarray, str]:
    key = BASELINE_FOR_TYPE[atype]
    signal_key = {"drift_flags": "cusum_signal", "jitter_flags": "jitter_signal",
                  "dtw_flags": "dtw_signal", "axis_flags": "axis_signal"}[key]
    signal = result[signal_key]
    name = BASELINE_NAME[key]
    return signal, name


def plot_overlay(atype: str, example: dict, output_dir: str) -> None:
    result = example["result"]
    label = example["label"]
    df = result["df"]
    starts = result["starts"]
    window_size = result["window_size"]
    window_centers = starts + window_size // 2
    window_ts = df["recorded_at_dt"].iloc[window_centers].values

    ts_start = df["recorded_at_dt"].iloc[label["start_idx"]]
    ts_end = df["recorded_at_dt"].iloc[min(label["end_idx"], len(df) - 1)]

    baseline_signal, baseline_name = _signal_for_type(atype, result)
    if baseline_signal is None:
        return
    # cusum_signal is per-SAMPLE (len(df)); the others are per-WINDOW (len(starts)).
    baseline_ts = df["recorded_at_dt"].values if len(baseline_signal) == len(df) else window_ts

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{atype} — device {example['device_id']} session {example['session_id']}", fontsize=13)

    for ax in axes:
        ax.axvspan(ts_start, ts_end, color="red", alpha=0.15, label="true injected interval")

    axes[0].plot(window_ts, result["lstm_errors"], color="tab:blue", label="LSTM reconstruction error")
    axes[0].axhline(result["threshold"], color="tab:blue", linestyle="--", alpha=0.6, label="LSTM threshold")
    axes[0].set_ylabel("Recon. error (MSE)")
    axes[0].legend(loc="upper right")

    axes[1].plot(baseline_ts, baseline_signal, color="tab:orange", label=f"{baseline_name} signal")
    axes[1].set_ylabel(baseline_name)
    axes[1].legend(loc="upper right")

    robot_df = example.get("robot_df")
    if robot_df is not None and len(robot_df):
        for joint_col in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"):
            if joint_col in robot_df.columns:
                axes[2].plot(robot_df["recorded_at_dt"], robot_df[joint_col], alpha=0.7, label=joint_col)
        axes[2].legend(loc="upper right", ncol=3, fontsize=8)
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Value (Degrees/Position)")
    axes[2].set_title("Robot joint ground truth (context — real motion timing, not an anomaly label)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"overlay_{atype}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM autoencoder vs. rule-based baselines")
    parser.add_argument("--model_dir", default="./out/models")
    parser.add_argument("--anomaly_dir", default="./out/anomaly_test_sets")
    parser.add_argument("--output_dir", default="./out/eval")
    parser.add_argument("--include_devices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    excluded = EXCLUDED_DEVICES_DEFAULT.copy()
    if args.include_devices:
        excluded -= set(args.include_devices)

    evaluate(args.model_dir, args.anomaly_dir, args.output_dir, excluded_devices=excluded)
