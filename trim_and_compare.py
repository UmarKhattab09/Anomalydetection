"""
Trim robot-joint and IMU-device logs to their "active" window and compare them.

Usage (per session):
    from trim_and_compare import process_session

    process_session(
        robot_csv="data/robot_session400.csv",
        imu_csv="data/imu_data_session400.csv",
        session_id="400",
        output_dir="out/session400",
    )

Or just run this file directly after editing the constants in `if __name__ == "__main__"`
at the bottom. Each joint is compared against its own device_id / channel(s), configured
in JOINT_DEVICE_MAP below.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# EDITABLE CONSTANTS — tune per device/session as needed
# --------------------------------------------------------------------------

# Activity threshold (degrees / position units) used to detect motion start/stop.
ACTIVITY_THRESHOLD_DEG: float = 1.0

# Extra samples kept before the detected start and after the detected end.
PADDING_SAMPLES: int = 5

# Which robot joint is compared against which IMU device/channel(s).
# - channels: one or more of "yaw", "pitch", "roll" recorded by that device.
# - invert: True flips the sign of every listed channel (some devices are mounted
#   opposite the joint's rotation convention).
JOINT_DEVICE_MAP: list[dict] = [
    {"joint": "joint_1", "device_id": 280, "channels": ["yaw"], "invert": False},
    {"joint": "joint_2", "device_id": 281, "channels": ["yaw", "pitch"], "invert": False},
    {"joint": "joint_3", "device_id": 282, "channels": ["yaw", "pitch"], "invert": True},
    {"joint": "joint_4", "device_id": 279, "channels": ["pitch"], "invert": False},
]

# Devices known to have a baseline offset that must be removed before thresholding.
# offset is subtracted from the named channel before anything else happens.
# e.g. {282: {"yaw": 10.4}} if a given device_id has a known baseline yaw offset.
KNOWN_BASELINE_OFFSETS: dict[int, dict[str, float]] = {}

# If a device isn't in KNOWN_BASELINE_OFFSETS but you still want auto baseline removal,
# set this True to subtract the median of the first N samples from every IMU channel.
AUTO_BASELINE_CORRECTION: bool = False
BASELINE_IDLE_SAMPLES: int = 20

# Tolerance for merge_asof time alignment (robot <-> IMU), in milliseconds.
MERGE_TOLERANCE_MS: int = 50


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------

def _to_datetime_auto(series: pd.Series) -> pd.Series:
    """Convert a recorded_at column to datetime, auto-detecting seconds vs ms vs us epoch."""
    numeric = pd.to_numeric(series, errors="coerce")
    magnitude = numeric.dropna().abs().median()
    if magnitude > 1e14:
        unit = "us"
    elif magnitude > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(numeric, unit=unit).astype("datetime64[ns]")


def load_robot(path: str, session_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["session_id"].astype(str) == str(session_id)].copy()
    df["recorded_at_dt"] = _to_datetime_auto(df["recorded_at"])
    df = df.sort_values("frame_id").reset_index(drop=True)
    return df


def load_imu_all(path: str, session_id: str) -> pd.DataFrame:
    """Load the full IMU table for a session (all device_ids) — filtered per-device later."""
    df = pd.read_csv(path)
    df = df[df["session_id"].astype(str) == str(session_id)].copy()
    df["recorded_at_dt"] = _to_datetime_auto(df["recorded_at"])
    return df


def select_imu_device(imu_all: pd.DataFrame, device_id: int) -> pd.DataFrame:
    df = imu_all[imu_all["device_id"] == device_id].copy()
    df = df.sort_values("frame_id").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Baseline offset + inversion
# --------------------------------------------------------------------------

def apply_baseline_correction(imu_df: pd.DataFrame, device_id: int, channels: list[str]) -> pd.DataFrame:
    imu_df = imu_df.copy()
    offsets = KNOWN_BASELINE_OFFSETS.get(device_id, {})
    for col in channels:
        if col not in imu_df.columns:
            continue
        if col in offsets:
            imu_df[col] = imu_df[col] - offsets[col]
        elif AUTO_BASELINE_CORRECTION:
            idle_median = imu_df[col].iloc[:BASELINE_IDLE_SAMPLES].median()
            imu_df[col] = imu_df[col] - idle_median
    return imu_df


def apply_inversion(imu_df: pd.DataFrame, channels: list[str], invert: bool) -> pd.DataFrame:
    if not invert:
        return imu_df
    imu_df = imu_df.copy()
    for col in channels:
        if col in imu_df.columns:
            imu_df[col] = -imu_df[col]
    return imu_df


# --------------------------------------------------------------------------
# Threshold-based active-window detection
# --------------------------------------------------------------------------

@dataclass
class ActiveWindow:
    column: str
    start_idx: int | None
    end_idx: int | None
    n_total: int

    @property
    def found(self) -> bool:
        return self.start_idx is not None and self.end_idx is not None


def find_active_window(series: pd.Series, threshold: float, padding: int) -> ActiveWindow:
    active = series.abs() > threshold
    active_idx = np.flatnonzero(active.to_numpy())

    if active_idx.size == 0:
        return ActiveWindow(column=series.name, start_idx=None, end_idx=None, n_total=len(series))

    start = max(0, int(active_idx[0]) - padding)
    end = min(len(series) - 1, int(active_idx[-1]) + padding)
    return ActiveWindow(column=series.name, start_idx=start, end_idx=end, n_total=len(series))


def table_active_window(df: pd.DataFrame, columns: list[str], threshold: float, padding: int) -> tuple[int | None, int | None, dict[str, ActiveWindow]]:
    """Union active window across all given columns (widest span needed to keep every channel's activity)."""
    per_column = {}
    starts, ends = [], []
    for col in columns:
        if col not in df.columns:
            continue
        win = find_active_window(df[col], threshold, padding)
        per_column[col] = win
        if win.found:
            starts.append(win.start_idx)
            ends.append(win.end_idx)

    if not starts:
        return None, None, per_column

    return min(starts), max(ends), per_column


# --------------------------------------------------------------------------
# Clock-based overlap (cross-check)
# --------------------------------------------------------------------------

def clock_overlap(robot_df: pd.DataFrame, imu_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(robot_df["recorded_at_dt"].min(), imu_df["recorded_at_dt"].min())
    end = min(robot_df["recorded_at_dt"].max(), imu_df["recorded_at_dt"].max())
    return start, end


# --------------------------------------------------------------------------
# Plotting — combined grid, one subplot per joint/device pair
# --------------------------------------------------------------------------

CHANNEL_LABEL = {"yaw": "Yaw", "pitch": "Pitch", "roll": "Roll"}
CHANNEL_COLOR = {"yaw": "tab:orange", "pitch": "tab:green", "roll": "tab:red"}


def _grid_shape(n: int) -> tuple[int, int]:
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    return rows, cols


def plot_combined_before(robot_raw: pd.DataFrame, imu_devices: dict[int, pd.DataFrame], session_id: str, out_path: str) -> None:
    n = len(JOINT_DEVICE_MAP)
    rows, cols = _grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for i, pair in enumerate(JOINT_DEVICE_MAP):
        ax = axes[i // cols][i % cols]
        joint_col, device_id, channels, invert = pair["joint"], pair["device_id"], pair["channels"], pair["invert"]
        imu_df = imu_devices[device_id]

        ax.plot(robot_raw["frame_id"], robot_raw[joint_col], label=f"Robot {joint_col.replace('_', ' ').title()}", color="tab:blue")
        for ch in channels:
            label = f"Device {device_id} {'Inverted ' if invert else ''}{CHANNEL_LABEL.get(ch, ch)}"
            ax.plot(imu_df["frame_id"], imu_df[ch], label=label, color=CHANNEL_COLOR.get(ch), linestyle="--" if invert else "-")

        title_channels = " & ".join(CHANNEL_LABEL.get(c, c) for c in channels)
        prefix = "Inverted " if invert else ""
        ax.set_title(f"{joint_col.replace('_', ' ').title()} vs Device {device_id} {prefix}{title_channels}")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Value (Degrees/Position)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Combined Robot Joint and IMU Device Data (Session {session_id})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_after(merged_pairs: dict[str, pd.DataFrame], session_id: str, out_path: str) -> None:
    n = len(JOINT_DEVICE_MAP)
    rows, cols = _grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for i, pair in enumerate(JOINT_DEVICE_MAP):
        ax = axes[i // cols][i % cols]
        joint_col, device_id, channels, invert = pair["joint"], pair["device_id"], pair["channels"], pair["invert"]
        merged = merged_pairs[joint_col]

        ax.plot(merged["recorded_at_dt"], merged[joint_col], label=f"Robot {joint_col.replace('_', ' ').title()}", color="tab:blue")
        for ch in channels:
            label = f"Device {device_id} {'Inverted ' if invert else ''}{CHANNEL_LABEL.get(ch, ch)}"
            ax.plot(merged["recorded_at_dt"], merged[ch], label=label, color=CHANNEL_COLOR.get(ch), linestyle="--" if invert else "-")

        title_channels = " & ".join(CHANNEL_LABEL.get(c, c) for c in channels)
        prefix = "Inverted " if invert else ""
        ax.set_title(f"{joint_col.replace('_', ' ').title()} vs Device {device_id} {prefix}{title_channels} (trimmed & aligned)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value (Degrees/Position)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Combined Robot Joint and IMU Device Data — Trimmed (Session {session_id})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

_TRIMMED_IMU_COLS: list[str] = [
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z", "yaw", "pitch", "roll",
]


def process_session(robot_csv: str, imu_csv: str, session_id: str, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    robot_raw = load_robot(robot_csv, session_id)
    imu_all = load_imu_all(imu_csv, session_id)

    imu_devices: dict[int, pd.DataFrame] = {}
    merged_pairs: dict[str, pd.DataFrame] = {}
    imu_trimmed_by_device: dict[int, pd.DataFrame] = {}
    summary_rows = []

    for pair in JOINT_DEVICE_MAP:
        joint_col, device_id, channels, invert = pair["joint"], pair["device_id"], pair["channels"], pair["invert"]

        imu_df = select_imu_device(imu_all, device_id)
        imu_df = apply_baseline_correction(imu_df, device_id, channels)
        imu_df = apply_inversion(imu_df, channels, invert)
        imu_devices[device_id] = imu_df

        # threshold-based windows
        robot_win = find_active_window(robot_raw[joint_col], ACTIVITY_THRESHOLD_DEG, PADDING_SAMPLES)
        imu_start_idx, imu_end_idx, imu_channel_windows = table_active_window(
            imu_df, channels, ACTIVITY_THRESHOLD_DEG, PADDING_SAMPLES
        )

        robot_trimmed = (
            robot_raw.iloc[robot_win.start_idx:robot_win.end_idx + 1].reset_index(drop=True)
            if robot_win.found else robot_raw.iloc[0:0]
        )
        imu_trimmed = (
            imu_df.iloc[imu_start_idx:imu_end_idx + 1].reset_index(drop=True)
            if imu_start_idx is not None else imu_df.iloc[0:0]
        )

        imu_trimmed_by_device.setdefault(device_id, imu_trimmed)  # joint_2/3 share a device; keep the first window

        clock_start, clock_end = clock_overlap(robot_raw, imu_df)

        merged = pd.merge_asof(
            robot_trimmed.sort_values("recorded_at_dt"),
            imu_trimmed.sort_values("recorded_at_dt"),
            on="recorded_at_dt",
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS),
            suffixes=("_robot", "_imu"),
        )
        merged = merged.dropna(subset=channels)
        merged_pairs[joint_col] = merged

        print(f"=== {joint_col} vs device {device_id} ({', '.join(channels)}{' inverted' if invert else ''}) ===")
        print(f"Threshold-based robot window: idx [{robot_win.start_idx}, {robot_win.end_idx}] of {len(robot_raw)} rows")
        print(f"Threshold-based IMU window:   idx [{imu_start_idx}, {imu_end_idx}] of {len(imu_df)} rows")
        print(f"Clock-based overlap:          {clock_start} -> {clock_end}")
        print()

        for ch in channels:
            ch_win = imu_channel_windows.get(ch)
            summary_rows.append({
                "joint": joint_col,
                "device_id": device_id,
                "imu_channel": ch,
                "inverted": invert,
                "robot_rows_orig": len(robot_raw),
                "robot_rows_trimmed": len(robot_trimmed),
                "imu_rows_orig": len(imu_df),
                "imu_rows_trimmed": len(imu_trimmed),
                "merged_rows": len(merged),
                "joint_threshold_start_idx": robot_win.start_idx,
                "joint_threshold_end_idx": robot_win.end_idx,
                "imu_threshold_start_idx": ch_win.start_idx if ch_win else None,
                "imu_threshold_end_idx": ch_win.end_idx if ch_win else None,
                "clock_start": clock_start,
                "clock_end": clock_end,
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    before_path = os.path.join(output_dir, f"before_session{session_id}.png")
    after_path = os.path.join(output_dir, f"after_session{session_id}.png")
    plot_combined_before(robot_raw, imu_devices, session_id, before_path)
    plot_combined_after(merged_pairs, session_id, after_path)

    summary_csv = os.path.join(output_dir, f"summary_session{session_id}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")

    # CSVs of the movement itself, trimmed to the active window (movement start -> end).
    #  trimmed_session<id>.csv      data behind the "after" plot: robot joints + IMU readings
    #                               matched at the robot's sample rate, one block per joint/device pair
    #  trimmed_imu_session<id>.csv  the same trimmed IMU windows at the IMU's native sample rate
    #                               (this is the signal the autoencoders see)
    device_by_joint = {pair["joint"]: pair["device_id"] for pair in JOINT_DEVICE_MAP}
    imu_cols = [c for c in _TRIMMED_IMU_COLS]
    robot_cols = [f"joint_{i}" for i in range(1, 7)]

    blocks = []
    for joint_col, merged in merged_pairs.items():
        block = merged[["recorded_at_dt"] + [c for c in robot_cols if c in merged.columns]
                       + [c for c in imu_cols if c in merged.columns]].copy()
        block.insert(0, "device_id", device_by_joint[joint_col])
        block.insert(0, "joint", joint_col)
        blocks.append(block)
    trimmed_csv = os.path.join(output_dir, f"trimmed_session{session_id}.csv")
    pd.concat(blocks, ignore_index=True).to_csv(trimmed_csv, index=False)

    imu_blocks = []
    for device_id, imu_trim in imu_trimmed_by_device.items():
        imu_blocks.append(imu_trim[["device_id", "recorded_at_dt"] + [c for c in imu_cols if c in imu_trim.columns]])
    trimmed_imu_csv = os.path.join(output_dir, f"trimmed_imu_session{session_id}.csv")
    pd.concat(imu_blocks, ignore_index=True).to_csv(trimmed_imu_csv, index=False)
    print(f"Saved trimmed IMU data to {trimmed_imu_csv}")
    print(f"Saved trimmed movement data to {trimmed_csv}")
    print(f"Saved plots to {before_path} and {after_path}")

    return summary_df


# --------------------------------------------------------------------------
# Batch entry point — auto-discovers session files, so adding a new session's
# CSVs to DATA_DIR and re-running `python trim_and_compare.py` just works.
# --------------------------------------------------------------------------

DATA_DIR: str = "data"
OUTPUT_ROOT: str = "out"

# Matches any of: robot_data_session398.csv, robot_session400.csv, "robot session 402.csv"
_ROBOT_PATTERN = re.compile(r"robot[ _]?(?:data[ _])?session[ _]?(\d+)", re.IGNORECASE)
# Matches any of: imu_data_session398.csv, "IMU session 402.csv"
_IMU_PATTERN = re.compile(r"imu[ _]?(?:data[ _])?session[ _]?(\d+)", re.IGNORECASE)


def discover_sessions(data_dir: str = DATA_DIR) -> list[dict]:
    """Scan data_dir for robot/IMU CSVs and pair them up by session id found in the filename."""
    robot_files: dict[str, str] = {}
    imu_files: dict[str, str] = {}

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)

        robot_match = _ROBOT_PATTERN.search(fname)
        imu_match = _IMU_PATTERN.search(fname)

        # a filename like "robot session 402.csv" won't match _IMU_PATTERN and vice versa,
        # but check imu first since "imu_data_session..." never matches _ROBOT_PATTERN either.
        if imu_match and "imu" in fname.lower():
            imu_files[imu_match.group(1)] = path
        elif robot_match and "robot" in fname.lower():
            robot_files[robot_match.group(1)] = path

    sessions = []
    for sid in sorted(set(robot_files) & set(imu_files), key=int):
        sessions.append({"session_id": sid, "robot_csv": robot_files[sid], "imu_csv": imu_files[sid]})

    missing_imu = sorted(set(robot_files) - set(imu_files), key=int)
    missing_robot = sorted(set(imu_files) - set(robot_files), key=int)
    if missing_imu:
        print(f"Skipping session(s) with robot data but no matching IMU file: {missing_imu}")
    if missing_robot:
        print(f"Skipping session(s) with IMU data but no matching robot file: {missing_robot}")

    return sessions


def run_all_sessions(sessions: list[dict] | None = None, output_root: str = OUTPUT_ROOT) -> dict[str, pd.DataFrame]:
    """Run process_session for every discovered (or given) session, continuing past per-session failures."""
    if sessions is None:
        sessions = discover_sessions()

    results = {}
    for entry in sessions:
        sid = entry["session_id"]
        print(f"\n{'=' * 70}\nSession {sid}\n{'=' * 70}")
        try:
            results[sid] = process_session(
                robot_csv=entry["robot_csv"],
                imu_csv=entry["imu_csv"],
                session_id=sid,
                output_dir=os.path.join(output_root, f"session{sid}"),
            )
        except Exception as exc:
            print(f"!!! Session {sid} failed: {exc}")
    return results


if __name__ == "__main__":
    # Auto-discovers every robot/IMU CSV pair under DATA_DIR and processes each one.
    # Drop a new session's two CSVs into DATA_DIR and just re-run this file.
    run_all_sessions()

    # Or run a single session instead:
    # process_session(
    #     robot_csv="data/robot_session400.csv",
    #     imu_csv="data/imu_data_session400.csv",
    #     session_id="400",
    #     output_dir="out/session400",
    # )
