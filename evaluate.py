"""
IMU Anomaly Detection — Evaluation (autoencoders vs. rule-based baselines)
==========================================================================
Scores the trained per-device autoencoders (dense_ae, lstm_ae, rnn_ae) and the rule-based
baselines (rule_baselines.py) against the SAME held-out, anomaly-injected test
sessions (generate_anomalies.py output), using the injected ground-truth
intervals to compute precision/recall/F1 PER ANOMALY TYPE — not one aggregate
normal-vs-anomaly number.

Usage
-----
  python train_imu_autoencoder.py                                # 1. train -> models/trained/<model>/
  python generate_anomalies.py --output_dir ./out/anomaly_test_sets  # 2. inject
  python evaluate.py --model_dir ./models/trained --anomaly_dir ./out/anomaly_test_sets --output_dir ./out/eval

Outputs (in --output_dir)
--------------------------
  device_summary.csv       per model and device: sample rate, session counts, threshold, aggregate test F1
  comparison_table.csv     per anomaly type: each model's P/R/F1 vs. its rule-based baseline P/R/F1
  overlay_<type>.png       one example per anomaly type: every model's error + baseline signal +
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
from models.autoencoder import MODEL_CLASSES, load_model_for_device
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
from scoring import two_sided_bounds, two_sided_flags, val_reference_errors

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

def score_session(csv_path: str, labels_path: str, models: dict[str, tuple], normalizer: ZScoreNormalizer,
                   reference_window: np.ndarray, reference_pool: np.ndarray, torch_device: torch.device,
                   cusum_h: float | None = None, ref_errors: dict[str, np.ndarray] | None = None,
                   score_mode: str = "two_sided") -> dict:
    """models: {model_name: (model, meta)} for one device. Channels/window/normalizer are
    identical across a device's models (same split, same train sessions), so the first
    model's meta drives windowing; each model keeps its own thresholds.
    score_mode "two_sided" flags errors outside the central 95% of clean val error (ref_errors,
    per model); "one_sided" flags errors above the 95th-percentile threshold. See scoring.py."""
    df = pd.read_csv(csv_path)
    df["recorded_at_dt"] = pd.to_datetime(df["recorded_at_dt"])  # CSV round-trip loses the dtype
    with open(labels_path) as f:
        labels = json.load(f)

    meta = next(iter(models.values()))[1]
    channels = meta["channels"]
    window_size = meta["window_size"]
    window_step = meta["window_step"]

    arr_norm = normalizer.transform(df[channels].values.astype(np.float64))
    windows, starts = make_windows(arr_norm, window_size, window_step)
    if len(windows) == 0:
        return {"error": "no windows"}

    model_scores = {}
    window_tensor = torch.tensor(windows, dtype=torch.float32).to(torch_device)
    with torch.no_grad():
        for name, (model, m_meta) in models.items():
            errors = model.reconstruction_error(window_tensor).cpu().numpy()
            if score_mode == "two_sided":
                lo, hi = two_sided_bounds(ref_errors[name])
                model_scores[name] = {"errors": errors, "flags": two_sided_flags(errors, (lo, hi)),
                                      "threshold": hi, "lower": lo}
            else:
                model_scores[name] = {"errors": errors, "flags": errors > m_meta["threshold"],
                                      "threshold": m_meta["threshold"], "lower": None}

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
        "model_scores": model_scores,
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

def evaluate(model_dir: str, anomaly_dir: str, output_dir: str, excluded_devices: set[int] | None = None,
             model_names: list[str] | None = None, score_mode: str = "two_sided") -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_names is None:
        model_names = [n for n in sorted(MODEL_CLASSES) if os.path.isdir(os.path.join(model_dir, n))]
    if not model_names:
        raise SystemExit(f"No trained models found under {model_dir} (expected subfolders {sorted(MODEL_CLASSES)})")
    print(f"Evaluating models: {model_names}")

    with open(os.path.join(anomaly_dir, "manifest.json")) as f:
        manifest = json.load(f)

    by_device: dict[int, list[dict]] = {}
    for entry in manifest:
        by_device.setdefault(entry["device_id"], []).append(entry)

    # Per-type accumulators: per model, and once for the (model-independent) baselines
    per_type_model: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        n: {t: [] for t in ANOMALY_TYPES} for n in model_names}
    per_type_baseline: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {t: [] for t in ANOMALY_TYPES}

    device_summary_rows = []
    example_for_type: dict[str, dict] = {}
    model_f1s: dict[tuple[str, int], list[float]] = {}

    device_sessions = load_all_device_sessions(excluded_devices=excluded_devices)
    splits = session_level_split(device_sessions)

    for device_id, entries in by_device.items():
        models: dict[str, tuple] = {}
        for name in model_names:
            loaded = load_model_for_device(device_id, os.path.join(model_dir, name), torch_device)
            if loaded is None:
                print(f"  [skip] device {device_id}: no trained {name} found in {model_dir}")
                continue
            models[name] = loaded
        if not models:
            continue
        meta = next(iter(models.values()))[1]
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
        ref_errors = {name: val_reference_errors(m, normalizer, splits[device_id].val, meta["channels"],
                                                 meta["window_size"], meta["window_step"])
                      for name, (m, _) in models.items()}

        primary_col = _reference_channel(meta["channels"])
        train_primary_series = np.concatenate([s.df[primary_col].values.astype(np.float64) for s in train_sessions]) if train_sessions else np.array([])
        cusum_h = calibrate_cusum_h(train_primary_series) if len(train_primary_series) else None

        for entry in entries:
            if reference_window is None:
                continue
            result = score_session(entry["csv"], entry["labels"], models, normalizer,
                                    reference_window, train_windows, torch_device, cusum_h=cusum_h,
                                    ref_errors=ref_errors, score_mode=score_mode)
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

                for name, score in result["model_scores"].items():
                    y_pred = score["flags"][eval_mask]
                    per_type_model[name][atype].append((y_true, y_pred))
                    _, _, f1 = precision_recall_f1(y_true, y_pred)
                    model_f1s.setdefault((name, device_id), []).append(f1)

                baseline_flags = result.get(BASELINE_FOR_TYPE[atype])
                if baseline_flags is not None:
                    per_type_baseline[atype].append((y_true, baseline_flags[eval_mask]))

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

        for name, (_, m_meta) in models.items():
            f1s = model_f1s.get((name, device_id), [])
            lo_hi = two_sided_bounds(ref_errors[name])
            device_summary_rows.append({
                "model": name,
                "device_id": device_id,
                "median_effective_hz": round(rate_df["effective_hz"].median(), 2) if len(rate_df) else None,
                "n_train_sessions": len(splits[device_id].train),
                "n_val_sessions": len(splits[device_id].val),
                "n_test_sessions": len(splits[device_id].test),
                "score_mode": score_mode,
                "threshold_upper": round(lo_hi[1] if score_mode == "two_sided" else m_meta["threshold"], 6),
                "threshold_lower": round(lo_hi[0], 6) if score_mode == "two_sided" else None,
                "mean_test_f1": round(float(np.mean(f1s)), 4) if f1s else None,
            })

    # ── Comparison table: per anomaly type, every model vs. its rule-based baseline ──
    comparison_rows = []
    for atype in ANOMALY_TYPES:
        row = {"anomaly_type": atype}
        n_eval = 0
        for name in model_names:
            pairs = per_type_model[name][atype]
            if pairs:
                y_true = np.concatenate([p[0] for p in pairs])
                y_pred = np.concatenate([p[1] for p in pairs])
                mp, mr, mf = precision_recall_f1(y_true, y_pred)
                n_eval = len(y_true)
            else:
                mp = mr = mf = None
            row[f"{name}_precision"], row[f"{name}_recall"], row[f"{name}_f1"] = mp, mr, mf
        row["n_eval_windows"] = n_eval

        baseline_pairs = per_type_baseline[atype]
        row["baseline"] = BASELINE_NAME[BASELINE_FOR_TYPE[atype]]
        if baseline_pairs:
            y_true_b = np.concatenate([p[0] for p in baseline_pairs])
            y_pred_b = np.concatenate([p[1] for p in baseline_pairs])
            row["baseline_precision"], row["baseline_recall"], row["baseline_f1"] = precision_recall_f1(y_true_b, y_pred_b)
        else:
            row["baseline_precision"] = row["baseline_recall"] = row["baseline_f1"] = None
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    summary_df = pd.DataFrame(device_summary_rows)

    print(f"\n{'=' * 70}\n Per-device summary\n{'=' * 70}")
    print(summary_df.to_string(index=False))
    f1_cols = ["anomaly_type", "n_eval_windows"] + [f"{n}_f1" for n in model_names] + ["baseline", "baseline_f1"]
    print(f"\n{'=' * 70}\n F1 per anomaly type: models vs. rule-based baseline\n{'=' * 70}")
    print(comparison_df[f1_cols].to_string(index=False))

    summary_path = os.path.join(output_dir, "device_summary.csv")
    comparison_path = os.path.join(output_dir, "comparison_table.csv")
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved: {summary_path}\nSaved: {comparison_path}")

    # ── Overlay plots — one example per anomaly type, all models on the error panel ──
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

    # Models have different error scales, so plot error / threshold: 1.0 is each model's own cutoff.
    for name, score in result["model_scores"].items():
        axes[0].plot(window_ts, score["errors"] / score["threshold"], label=f"{name} error / threshold")
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.6, label="upper threshold")
    first = next(iter(result["model_scores"].values()))
    if first.get("lower") is not None:
        axes[0].axhline(first["lower"] / first["threshold"], color="dimgray", linestyle=":", alpha=0.8,
                        label=f"lower threshold ({next(iter(result['model_scores']))})")
    axes[0].set_ylabel("Recon. error / threshold")
    axes[0].legend(loc="upper right", fontsize=8)

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
    parser = argparse.ArgumentParser(description="Evaluate autoencoders (dense_ae, lstm_ae, rnn_ae) vs. rule-based baselines")
    parser.add_argument("--model_dir", default="./models/trained", help="Folder holding one subfolder per trained model")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CLASSES), default=None,
                        help="Which models to evaluate (default: every model with a trained subfolder)")
    parser.add_argument("--anomaly_dir", default="./out/anomaly_test_sets")
    parser.add_argument("--output_dir", default="./out/eval")
    parser.add_argument("--include_devices", type=int, nargs="+", default=None)
    parser.add_argument("--score_mode", choices=["two_sided", "one_sided"], default="two_sided",
                        help="How model flags are decided (see scoring.py); baselines are unaffected")
    args = parser.parse_args()

    excluded = EXCLUDED_DEVICES_DEFAULT.copy()
    if args.include_devices:
        excluded -= set(args.include_devices)

    evaluate(args.model_dir, args.anomaly_dir, args.output_dir, excluded_devices=excluded, model_names=args.models,
             score_mode=args.score_mode)
