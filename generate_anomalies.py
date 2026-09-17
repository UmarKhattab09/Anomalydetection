"""
IMU Anomaly Data Generator (per-session, localized injection)
===============================================================
Injects labelled synthetic anomalies of each of the 7 taxonomy types into the
HELD-OUT TEST sessions only (never train/val — that would leak anomaly shapes
into what the autoencoder learns as "normal"). Each injection is localized to a
random sub-interval of the session and its exact [start_idx, end_idx) is recorded,
so evaluate.py can score detection against real ground truth instead of guessing.

Unlike thesiswork/scripts/generate_anomalies.py (which labels an entire fixed-length
motion segment as one anomaly type), this generalizes to full continuous sessions
covering ALL available IMU channels (accel/gyro/mag/yaw/pitch/roll — whichever a
device reports), with anomaly magnitude scaled relative to each channel's own local
variability so the same injection code works whether the channel is degrees or g's.

Usage
-----
  python generate_anomalies.py --output_dir ./out/anomaly_test_sets

Output structure
----------------
  out/anomaly_test_sets/
    device{ID}_session{SID}_anomalies.csv    <- full session, IMU channels + anomaly_type column
    device{ID}_session{SID}_labels.json      <- [{type, start_idx, end_idx}, ...] ground truth
    manifest.json                             <- every generated file + its device/session/types

Requirements: pip install pandas numpy scipy
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from pipeline_common import (
    ALL_IMU_CHANNELS,
    DeviceSession,
    EXCLUDED_DEVICES_DEFAULT,
    WINDOW_SIZE,
    load_all_device_sessions,
    session_level_split,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ANOMALY_TYPES: list[str] = [
    "overshoot", "undershoot", "stall", "jitter", "drift", "early_stop", "axis_decouple",
]

# Injected anomaly interval length range, in samples (~1-3s at 100Hz).
SEGMENT_LEN_RANGE: tuple[int, int] = (100, 300)

# Minimum gap (samples) kept between consecutive injected intervals so effects
# don't bleed into each other and confuse ground-truth labeling.
MIN_GAP_SAMPLES: int = 50

SMOOTH_SIGMA: float = 1.5

SEED = 42


@dataclass
class AnomalyLabel:
    type: str
    start_idx: int
    end_idx: int  # exclusive


# --------------------------------------------------------------------------
# Per-type injection — operates on an arbitrary channel list, amplitude scaled
# relative to each channel's own local std so it generalizes across accel/gyro/
# mag/yaw/pitch/roll without hand-tuning per channel.
# --------------------------------------------------------------------------

def _local_std(df: pd.DataFrame, col: str) -> float:
    s = df[col].std()
    return float(s) if s > 1e-9 else 1.0


def _reference_channel(channels: list[str]) -> str:
    for preferred in ("yaw", "gyro_z", "pitch", "gyro_y", "roll"):
        if preferred in channels:
            return preferred
    return channels[0]


def apply_anomaly(df: pd.DataFrame, start: int, end: int, channels: list[str],
                   anomaly_type: str, rng: np.random.Generator) -> pd.DataFrame:
    """Apply one anomaly type to df.iloc[start:end] for the given channels, in place on a copy."""
    out = df.copy()
    seg_len = end - start
    ref_col = _reference_channel(channels)

    if anomaly_type == "overshoot":
        scale = rng.uniform(1.10, 1.30)
        taper = 1.0 - np.abs(np.linspace(-1, 1, seg_len))  # peak in the middle, tapers at edges
        factor = 1.0 + (scale - 1.0) * taper
        for col in channels:
            out.iloc[start:end, out.columns.get_loc(col)] = df[col].iloc[start:end].values * factor

    elif anomaly_type == "undershoot":
        scale = rng.uniform(0.60, 0.85)
        for col in channels:
            out.iloc[start:end, out.columns.get_loc(col)] = df[col].iloc[start:end].values * scale

    elif anomaly_type == "stall":
        for col in channels:
            freeze_val = df[col].iloc[start]
            out.iloc[start:end, out.columns.get_loc(col)] = freeze_val

    elif anomaly_type == "jitter":
        for col in channels:
            noise_std = _local_std(df, col) * rng.uniform(1.5, 4.0)
            noise = rng.normal(0, noise_std, size=seg_len)
            out.iloc[start:end, out.columns.get_loc(col)] = df[col].iloc[start:end].values + noise

    elif anomaly_type == "drift":
        # Ramp up then back down within the interval (triangular), so the bias
        # creep is confined to this injected interval instead of permanently
        # offsetting every later sample — a real un-recovered drift would do that,
        # but here it would silently contaminate every OTHER anomaly type injected
        # later in the same session (their windows would inherit the leftover
        # offset and no longer look like a clean example of their own type).
        drift_dir = rng.choice([-1, 1])
        half = seg_len // 2
        triangle = np.concatenate([np.arange(half), np.arange(seg_len - half, 0, -1)])
        for col in channels:
            slope = _local_std(df, col) * rng.uniform(0.02, 0.08) * drift_dir
            ramp = triangle * slope
            out.iloc[start:end, out.columns.get_loc(col)] = df[col].iloc[start:end].values + ramp

    elif anomaly_type == "early_stop":
        # Reverse before completing the motion: mirror the first half of the
        # interval back toward baseline instead of continuing to whatever the
        # real trajectory does in the second half.
        reversal = start + seg_len // 3
        for col in channels:
            ascending = df[col].iloc[start:reversal].values
            remaining = end - reversal
            if len(ascending) > 0:
                descent_template = ascending[::-1]
                repeats = int(np.ceil(remaining / max(1, len(descent_template))))
                descent = np.tile(descent_template, repeats)[:remaining]
            else:
                descent = np.full(remaining, df[col].iloc[start])
            out.iloc[start:reversal, out.columns.get_loc(col)] = ascending
            out.iloc[reversal:end, out.columns.get_loc(col)] = descent

    elif anomaly_type == "axis_decouple":
        # Reference channel stays normal; the OTHER channels develop an independent
        # spurious oscillation that has nothing to do with the reference motion —
        # this is what breaks cross-axis correlation (the rule baseline for this type).
        secondary = [c for c in channels if c != ref_col]
        for col in secondary:
            amp = _local_std(df, col) * rng.uniform(1.5, 3.5)
            freq = rng.uniform(0.03, 0.10)
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(seg_len)
            spurious = amp * np.sin(2 * np.pi * freq * t + phase)
            out.iloc[start:end, out.columns.get_loc(col)] = df[col].iloc[start:end].values + spurious

    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return out


def smooth_boundaries(df: pd.DataFrame, channels: list[str], start: int, end: int, sigma: float = SMOOTH_SIGMA,
                       pad: int = 10) -> pd.DataFrame:
    """Lightly smooth just around the injected boundaries to avoid unrealistically sharp edges."""
    out = df.copy()
    lo = max(0, start - pad)
    hi = min(len(out), end + pad)
    for col in channels:
        segment = out[col].iloc[lo:hi].values
        out.iloc[lo:hi, out.columns.get_loc(col)] = gaussian_filter1d(segment, sigma=sigma)
    return out


# --------------------------------------------------------------------------
# Per-session injection of all 7 types at non-overlapping intervals
# --------------------------------------------------------------------------

def _feasible_length_range(n_samples: int, n_intervals: int, min_gap: int,
                            default_range: tuple[int, int] = SEGMENT_LEN_RANGE) -> tuple[int, int]:
    """Shrink the segment length range so n_intervals of it plus gaps can actually fit in n_samples."""
    budget = n_samples - min_gap * (n_intervals + 1)
    max_len_that_fits = max(20, budget // n_intervals)
    lo, hi = default_range
    hi = min(hi, max_len_that_fits)
    lo = min(lo, max(20, hi // 2))
    if lo >= hi:
        lo = max(10, hi - 10)
    return lo, hi


def pick_non_overlapping_intervals(n_samples: int, n_intervals: int, rng: np.random.Generator,
                                    length_range: tuple[int, int] = SEGMENT_LEN_RANGE,
                                    min_gap: int = MIN_GAP_SAMPLES, max_tries: int = 300) -> list[tuple[int, int]]:
    length_range = _feasible_length_range(n_samples, n_intervals, min_gap, length_range)
    intervals: list[tuple[int, int]] = []
    for _ in range(n_intervals):
        for _try in range(max_tries):
            length = int(rng.integers(length_range[0], length_range[1] + 1))
            if length >= n_samples:
                break
            start = int(rng.integers(0, n_samples - length))
            end = start + length
            if all(end + min_gap <= s or start - min_gap >= e for s, e in intervals):
                intervals.append((start, end))
                break
    return sorted(intervals)


def inject_session_anomalies(session: DeviceSession, rng: np.random.Generator) -> tuple[pd.DataFrame, list[AnomalyLabel]]:
    df = session.df.copy()
    df["anomaly_type"] = "normal"
    n = len(df)

    intervals = pick_non_overlapping_intervals(n, len(ANOMALY_TYPES), rng)
    if len(intervals) < len(ANOMALY_TYPES):
        print(f"  [warn] device {session.device_id} session {session.session_id}: only fit "
              f"{len(intervals)}/{len(ANOMALY_TYPES)} non-overlapping anomaly intervals "
              f"(session has {n} samples) — some types skipped for this session")

    labels: list[AnomalyLabel] = []
    for anomaly_type, (start, end) in zip(ANOMALY_TYPES, intervals):
        df = apply_anomaly(df, start, end, session.channels, anomaly_type, rng)
        df = smooth_boundaries(df, session.channels, start, end)
        df.iloc[start:end, df.columns.get_loc("anomaly_type")] = anomaly_type
        labels.append(AnomalyLabel(type=anomaly_type, start_idx=start, end_idx=end))

    return df, labels


# --------------------------------------------------------------------------
# Main generator — runs over every device's TEST sessions (per the shared split)
# --------------------------------------------------------------------------

def generate(output_dir: str, excluded_devices: set[int] | None = None, seed: int = SEED) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    device_sessions = load_all_device_sessions(excluded_devices=excluded_devices)
    splits = session_level_split(device_sessions)

    manifest = []
    for device_id, split in splits.items():
        for session in split.test:
            rng = np.random.default_rng(seed + device_id + int(session.session_id))
            labeled_df, labels = inject_session_anomalies(session, rng)

            csv_path = os.path.join(output_dir, f"device{device_id}_session{session.session_id}_anomalies.csv")
            labels_path = os.path.join(output_dir, f"device{device_id}_session{session.session_id}_labels.json")

            labeled_df.to_csv(csv_path, index=False)
            with open(labels_path, "w") as f:
                json.dump([asdict(l) for l in labels], f, indent=2)

            manifest.append({
                "device_id": device_id, "session_id": session.session_id,
                "csv": csv_path, "labels": labels_path,
                "n_samples": len(labeled_df), "types_injected": [l.type for l in labels],
            })
            print(f"  device {device_id} session {session.session_id}: injected "
                  f"{len(labels)}/{len(ANOMALY_TYPES)} types -> {csv_path}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(manifest)} labeled test session(s) across {len(splits)} device(s).")
    print(f"Manifest: {manifest_path}")
    return {"manifest": manifest, "manifest_path": manifest_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject synthetic labeled anomalies into held-out test sessions")
    parser.add_argument("--output_dir", default="./out/anomaly_test_sets")
    parser.add_argument("--include_devices", type=int, nargs="+", default=None,
                         help="Override EXCLUDED_DEVICES_DEFAULT (e.g. --include_devices 280 281 282 283) "
                              "to force-include a normally-excluded device such as 280.")
    args = parser.parse_args()

    excluded = EXCLUDED_DEVICES_DEFAULT.copy()
    if args.include_devices:
        excluded -= set(args.include_devices)

    generate(output_dir=args.output_dir, excluded_devices=excluded)
