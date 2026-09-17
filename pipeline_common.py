"""
Shared session loading, sampling-rate verification, session-level splitting, and
z-score normalization for the IMU anomaly detection pipeline.

Single source of truth for anything that reads sessions from data/ — imported by
generate_anomalies.py, train_imu_autoencoder.py, evaluate.py, and rule_baselines.py
so all of them see the same sessions, channels, and normalization.

Reuses trim_and_compare.py's session discovery (regex-based robot/IMU filename
pairing) so adding a new session's two CSVs to DATA_DIR is enough for every script
in this pipeline, not just the plotting one.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trim_and_compare import (
    DATA_DIR,
    _to_datetime_auto,
    discover_sessions,
    find_active_window,
    load_robot,
    select_imu_device,
    table_active_window,
)

# --------------------------------------------------------------------------
# EDITABLE CONSTANTS
# --------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Every IMU channel a device might report. Only the ones actually present and
# non-constant in a given device's data are used — this is what lets the same
# code run whether a device reports 3 channels or all 12.
ALL_IMU_CHANNELS: list[str] = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "yaw", "pitch", "roll",
]

# Robot joint columns — validation-only ground truth, NEVER a model input.
ROBOT_JOINT_COLS: list[str] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Expected effective sampling rate for this dataset generation (100 Hz sessions only —
# older ~65-67 Hz sessions are a different generation and must not be mixed in).
EXPECTED_HZ: float = 100.0
HZ_DEVIATION_TOLERANCE_PCT: float = 15.0  # flag sessions whose measured rate deviates more than this

# Device 280 has a known amplitude/shape mismatch vs. its robot joint even after
# trimming/alignment (see trim_and_compare.py plots). Excluded by default, but
# never silently — pass include_devices to override.
EXCLUDED_DEVICES_DEFAULT: set[int] = {280}

# Session-level split sizes PER DEVICE. Session-level (not window-level) splitting
# is what prevents leakage — windows from the same session are highly correlated,
# so splitting by window would let near-duplicate windows appear in both train and
# test and silently inflate every metric.
N_TEST_SESSIONS_PER_DEVICE: int = 2
N_VAL_SESSIONS_PER_DEVICE: int = 1

# Sliding window config for the autoencoder and rule baselines.
# ~77 samples / 15 step @ 100Hz ≈ 0.77s window / 0.15s stride — matches the prior
# 50/10 @ ~65Hz convention's real-time span. See NOTE below on per-anomaly-type fit.
WINDOW_SIZE: int = 77
WINDOW_STEP: int = 15

# NOTE on window size vs. anomaly taxonomy (flagged per your request):
# - jitter is a high-frequency phenomenon — a 0.77s window comfortably contains many
#   cycles of typical mechanical chatter, so 77/15 resolves it fine, if anything a
#   SHORTER window (e.g. 30-40 samples) would localize jitter onset more precisely.
# - drift is a slow, multi-second bias creep — a single 0.77s window looks almost
#   normal early in a drift; drift is only reliably caught by the SEQUENCE of rising
#   reconstruction error across consecutive overlapping windows, not any one window.
#   This is fine with 77/15 because of the small step (15) giving dense overlap, but
#   don't expect a single-window flag to catch drift near its onset — evaluate drift
#   detection at the multi-window / interval level, not a lone window.
# - overshoot/undershoot/stall/early_stop are shape distortions of a single motion
#   and typically span 100-300 samples (1-3s) at 100Hz — a 77-sample window only
#   sees a piece of them. Multiple consecutive windows will each be partially
#   anomalous; this is expected and handled by the interval-overlap labeling in
#   generate_anomalies.py / evaluate.py, not by growing the window.
# - axis_decouple needs at least one full spurious-oscillation cycle to detect via
#   cross-correlation; injected oscillation periods here are tuned to fit inside
#   a 77-sample window.
# Net recommendation: 77/15 is reasonable as a general-purpose window and matches
# your prior convention, but expect drift/shape-distortion detection to operate on
# runs of consecutive flagged windows rather than isolated ones — evaluate.py scores
# at the ground-truth-interval level for exactly this reason.


# --------------------------------------------------------------------------
# Sampling rate verification
# --------------------------------------------------------------------------

@dataclass
class SamplingRateReport:
    session_id: str
    device_id: int
    n_samples: int
    median_dt_s: float
    effective_hz: float
    deviation_pct: float
    flagged: bool


def verify_sampling_rate(recorded_at_dt: pd.Series, session_id: str, device_id: int,
                          expected_hz: float = EXPECTED_HZ,
                          tolerance_pct: float = HZ_DEVIATION_TOLERANCE_PCT) -> SamplingRateReport:
    deltas = recorded_at_dt.diff().dt.total_seconds().dropna()
    median_dt = float(deltas.median()) if len(deltas) else float("nan")
    effective_hz = 1.0 / median_dt if median_dt > 0 else float("nan")
    deviation_pct = abs(effective_hz - expected_hz) / expected_hz * 100 if effective_hz == effective_hz else float("inf")
    flagged = not (deviation_pct <= tolerance_pct)
    return SamplingRateReport(
        session_id=session_id, device_id=device_id, n_samples=len(recorded_at_dt),
        median_dt_s=median_dt, effective_hz=effective_hz,
        deviation_pct=deviation_pct, flagged=flagged,
    )


# --------------------------------------------------------------------------
# Session container
# --------------------------------------------------------------------------

@dataclass
class DeviceSession:
    session_id: str
    device_id: int
    df: pd.DataFrame                 # IMU rows for this device, sorted by frame_id, active-window trimmed
    channels: list[str]              # channels actually usable for this device
    robot_df: pd.DataFrame           # robot rows for the whole session, active-window trimmed (ground truth only)
    sample_rate: SamplingRateReport


def get_device_channels(imu_df: pd.DataFrame, min_std: float = 1e-9) -> list[str]:
    """Channels present and non-constant for this device — a flat/missing channel is dropped, not zero-filled."""
    usable = []
    for col in ALL_IMU_CHANNELS:
        if col not in imu_df.columns:
            continue
        series = imu_df[col].dropna()
        if len(series) == 0 or series.std() < min_std:
            continue
        usable.append(col)
    return usable


def load_all_device_sessions(data_dir: str = DATA_DIR,
                              excluded_devices: set[int] | None = None) -> dict[int, list[DeviceSession]]:
    """
    Discover every session, load + trim robot and IMU tables, and return
    {device_id: [DeviceSession, ...]} across all discovered sessions.
    """
    if excluded_devices is None:
        excluded_devices = EXCLUDED_DEVICES_DEFAULT

    sessions_meta = discover_sessions(data_dir)
    result: dict[int, list[DeviceSession]] = {}

    for entry in sessions_meta:
        sid = entry["session_id"]
        robot_raw = load_robot(entry["robot_csv"], sid)
        robot_start, robot_end, _ = table_active_window(robot_raw, ROBOT_JOINT_COLS, 1.0, 5)
        robot_trimmed = (
            robot_raw.iloc[robot_start:robot_end + 1].reset_index(drop=True)
            if robot_start is not None else robot_raw
        )

        imu_all = pd.read_csv(entry["imu_csv"])
        imu_all = imu_all[imu_all["session_id"].astype(str) == str(sid)].copy()
        imu_all["recorded_at_dt"] = _to_datetime_auto(imu_all["recorded_at"])

        for device_id in sorted(int(d) for d in imu_all["device_id"].dropna().unique()):
            if device_id in excluded_devices:
                continue

            imu_df = select_imu_device(imu_all, device_id)
            channels = get_device_channels(imu_df)
            if not channels:
                continue

            # Trim IMU to its own active window on whichever orientation channels it has
            # (falls back to gyro magnitude if no yaw/pitch/roll — still a real activity signal).
            threshold_cols = [c for c in ("yaw", "pitch", "roll") if c in channels] or channels
            imu_start, imu_end, _ = table_active_window(imu_df, threshold_cols, 1.0, 5)
            imu_trimmed = (
                imu_df.iloc[imu_start:imu_end + 1].reset_index(drop=True)
                if imu_start is not None else imu_df
            )
            if len(imu_trimmed) < WINDOW_SIZE:
                print(f"  [skip] session {sid} device {device_id}: only {len(imu_trimmed)} rows after trim, "
                      f"need >= {WINDOW_SIZE}")
                continue

            rate_report = verify_sampling_rate(imu_trimmed["recorded_at_dt"], sid, device_id)

            result.setdefault(device_id, []).append(DeviceSession(
                session_id=sid, device_id=device_id, df=imu_trimmed, channels=channels,
                robot_df=robot_trimmed, sample_rate=rate_report,
            ))

    return result


def print_sampling_rate_report(device_sessions: dict[int, list[DeviceSession]]) -> pd.DataFrame:
    rows = []
    for device_id, sessions in device_sessions.items():
        for s in sessions:
            r = s.sample_rate
            rows.append({
                "device_id": device_id, "session_id": r.session_id, "n_samples": r.n_samples,
                "median_dt_s": round(r.median_dt_s, 5), "effective_hz": round(r.effective_hz, 2),
                "deviation_pct": round(r.deviation_pct, 1), "flagged": r.flagged,
            })
    df = pd.DataFrame(rows).sort_values(["device_id", "session_id"])
    print(df.to_string(index=False))
    n_flagged = df["flagged"].sum()
    if n_flagged:
        print(f"\n[warn] {n_flagged} session/device combo(s) deviate > {HZ_DEVIATION_TOLERANCE_PCT}% "
              f"from {EXPECTED_HZ} Hz — inspect before training (mixed-rate data silently degrades a "
              f"fixed-window-size model).")
    return df


# --------------------------------------------------------------------------
# Session-level train/val/test split
# --------------------------------------------------------------------------

@dataclass
class DeviceSplit:
    device_id: int
    train: list[DeviceSession]
    val: list[DeviceSession]
    test: list[DeviceSession]


def session_level_split(device_sessions: dict[int, list[DeviceSession]],
                         n_test: int = N_TEST_SESSIONS_PER_DEVICE,
                         n_val: int = N_VAL_SESSIONS_PER_DEVICE,
                         seed: int = SEED) -> dict[int, DeviceSplit]:
    """
    Hold out whole sessions per device for val/test — never split within a session,
    since windows from the same session are correlated and would leak across splits.
    Small session counts (e.g. 6-7 today) mean the held-out set is a big fraction of
    the data; that's expected early on and shrinks in relative terms as sessions scale.
    """
    splits = {}
    for device_id, sessions in device_sessions.items():
        sessions_sorted = sorted(sessions, key=lambda s: s.session_id)
        rng = random.Random(seed + device_id)
        shuffled = sessions_sorted.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test_actual = min(n_test, max(0, n - 1))          # always leave >= 1 for train
        n_val_actual = min(n_val, max(0, n - n_test_actual - 1))

        test = shuffled[:n_test_actual]
        val = shuffled[n_test_actual:n_test_actual + n_val_actual]
        train = shuffled[n_test_actual + n_val_actual:]

        if n_test_actual < n_test or n_val_actual < n_val:
            print(f"  [note] device {device_id}: only {n} session(s) available — "
                  f"using {len(train)} train / {len(val)} val / {len(test)} test "
                  f"(requested {n - n_test - n_val if n - n_test - n_val > 0 else '?'} / {n_val} / {n_test})")

        splits[device_id] = DeviceSplit(device_id=device_id, train=train, val=val, test=test)
    return splits


# --------------------------------------------------------------------------
# Per-device z-score normalization — fit on TRAIN sessions only
# --------------------------------------------------------------------------

@dataclass
class ZScoreNormalizer:
    channels: list[str]
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    std: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, sessions: list[DeviceSession]) -> "ZScoreNormalizer":
        data = np.concatenate([s.df[self.channels].values.astype(np.float64) for s in sessions], axis=0)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std < 1e-9] = 1.0  # avoid divide-by-zero on a near-constant channel
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean

    def to_dict(self) -> dict:
        return {"channels": self.channels, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ZScoreNormalizer":
        return cls(channels=d["channels"], mean=np.array(d["mean"]), std=np.array(d["std"]))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ZScoreNormalizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def make_windows(arr: np.ndarray, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across a (n_samples, n_features) array.
    Returns (windows, start_indices): windows shape (n_windows, window_size, n_features).
    """
    n = len(arr)
    starts = list(range(0, max(0, n - window_size + 1), step))
    if not starts:
        return np.empty((0, window_size, arr.shape[1])), np.empty((0,), dtype=int)
    windows = np.stack([arr[s:s + window_size] for s in starts])
    return windows.astype(np.float32), np.array(starts, dtype=int)
