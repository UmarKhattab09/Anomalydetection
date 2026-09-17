"""
Rule-based baseline anomaly detectors — one simple, interpretable function per
anomaly type, so evaluate.py can argue empirically where the LSTM autoencoder
actually earns its complexity vs. where a cheap classical detector ties it.

Each detector has a `*_scores` function (continuous signal, for overlay plots)
and a `*_flags` function built on top of it (boolean, for precision/recall),
both aligned to the same window starts the LSTM is scored on (see
pipeline_common.make_windows), so LSTM vs. baseline comparisons are apples-to-apples.

Deliberately NOT combined into one rule engine — each function is independent
and only claims to detect the one anomaly type it targets.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# drift — CUSUM (cumulative sum control chart) on slow bias creep
# --------------------------------------------------------------------------

def cusum_stat(series: np.ndarray, k: float | None = None) -> np.ndarray:
    """Per-sample two-sided CUSUM magnitude (max of the positive/negative accumulators)."""
    diffs = np.diff(series, prepend=series[0])
    if k is None:
        k = 0.5 * diffs.std() if diffs.std() > 1e-9 else 1e-6

    pos, neg = 0.0, 0.0
    stat = np.zeros(len(series))
    for i, d in enumerate(diffs):
        pos = max(0.0, pos + d - k)
        neg = min(0.0, neg + d + k)
        stat[i] = max(pos, -neg)
    return stat


def calibrate_cusum_h(train_series: np.ndarray, k: float | None = None, percentile: float = 99.0,
                       margin: float = 1.2) -> float:
    """
    Calibrate the CUSUM decision threshold h from a device's own NORMAL training data
    (analogous to how the LSTM and DTW thresholds are calibrated on normal data) —
    a fixed multiple of the raw series' std is NOT safe here: real robot motion swings
    are large (yaw std ~100+ degrees), so a flat "5*std" bar sits far above anything a
    CUSUM statistic on a genuine bias-creep would ever reach, and the detector would
    never fire regardless of whether drift is present.
    """
    stat = cusum_stat(train_series, k)
    baseline = np.percentile(stat, percentile) if len(stat) else 1e-6
    return max(baseline * margin, 1e-6)


def cusum_drift_flags(series: np.ndarray, window_starts: np.ndarray, window_size: int,
                       k: float | None = None, h: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided CUSUM over the full series. k = allowed slack (defaults to 0.5*std of
    the series' first-difference — i.e. only sustained deviation beyond normal sample-
    to-sample noise accumulates), h = decision threshold. h MUST be calibrated on
    normal training data via calibrate_cusum_h — the caller-visible default here
    (5*std of THIS series) is only a last-resort fallback for callers with no
    training data available, and is too conservative for real motion data (see
    calibrate_cusum_h's docstring).
    A window is flagged if any sample inside it has an accumulated CUSUM statistic
    that has crossed h (drift is a creeping effect, so this fires partway through
    the true interval and keeps firing — matching the "runs of windows" note in
    pipeline_common.py, not a single isolated window).
    Returns (window_flags, per_sample_stat) — the latter is what overlay plots draw.
    """
    if h is None:
        h = 5.0 * (series.std() if series.std() > 1e-9 else 1e-6)
    stat = cusum_stat(series, k)
    flags = np.array([(stat[s:s + window_size] > h).any() for s in window_starts])
    return flags, stat


# --------------------------------------------------------------------------
# jitter — windowed variance / high-frequency energy threshold
# --------------------------------------------------------------------------

def jitter_energy_per_window(series: np.ndarray, window_starts: np.ndarray, window_size: int) -> np.ndarray:
    diffs = np.diff(series, prepend=series[0])
    return np.array([diffs[s:s + window_size].var() for s in window_starts])


def jitter_variance_flags(series: np.ndarray, window_starts: np.ndarray, window_size: int,
                           baseline_var: float | None = None, factor: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Flags a window if its high-frequency energy (variance of the first difference,
    i.e. sample-to-sample jumps) exceeds `factor` times a baseline level. baseline_var
    defaults to the median per-window high-freq energy across the whole series, which
    approximates "normal" chatter level without needing a separate reference signal.
    Returns (window_flags, window_energy) — the latter is what overlay plots draw.
    """
    window_energy = jitter_energy_per_window(series, window_starts, window_size)
    if baseline_var is None:
        baseline_var = np.median(window_energy) if len(window_energy) else 1e-9
    baseline_var = max(baseline_var, 1e-12)
    return window_energy > factor * baseline_var, window_energy


# --------------------------------------------------------------------------
# overshoot / undershoot / stall — DTW distance to a reference "normal" trajectory
# --------------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Plain O(n*m) dynamic-programming DTW distance (no external dep — windows here
    are ~77 samples, so this is fast enough without fastdtw/dtaidistance).
    """
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (a[i - 1] - b[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, m]))


def build_reference_window(train_windows: np.ndarray) -> np.ndarray:
    """
    A representative "normal" trajectory for DTW comparison: the training window
    whose length-normalized shape is closest to the mean of all training windows
    (closest to centroid, rather than an arithmetic mean that would blur the actual
    motion shape overshoot/undershoot/stall are defined relative to).
    """
    mean_shape = train_windows.mean(axis=0)
    dists = np.linalg.norm(train_windows - mean_shape, axis=(1, 2))
    return train_windows[np.argmin(dists)]


def dtw_distances_per_window(windows: np.ndarray, reference: np.ndarray, channel_idx: int = 0) -> np.ndarray:
    ref_series = reference[:, channel_idx]
    return np.array([dtw_distance(w[:, channel_idx], ref_series) for w in windows])


def dtw_shape_flags(windows: np.ndarray, reference: np.ndarray, channel_idx: int = 0,
                     threshold: float | None = None,
                     reference_pool: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Flags a window if its DTW distance to `reference` (on one representative channel)
    exceeds a threshold. threshold defaults to mean + 3*std of reference-vs-reference-pool
    DTW distances (i.e. calibrated on normal data's own shape variability).
    Returns (window_flags, window_distances) — the latter is what overlay plots draw.
    """
    distances = dtw_distances_per_window(windows, reference, channel_idx)

    if threshold is None:
        if reference_pool is None:
            threshold = np.inf  # no calibration data — caller must supply one
        else:
            calib = dtw_distances_per_window(reference_pool, reference, channel_idx)
            threshold = calib.mean() + 3 * calib.std()

    return distances > threshold, distances


# --------------------------------------------------------------------------
# axis_decouple — rolling cross-correlation between related axes
# --------------------------------------------------------------------------

def rolling_xcorr_values(series_a: np.ndarray, series_b: np.ndarray, window_starts: np.ndarray,
                          window_size: int) -> np.ndarray:
    values = np.zeros(len(window_starts))
    for i, s in enumerate(window_starts):
        a = series_a[s:s + window_size]
        b = series_b[s:s + window_size]
        if a.std() < 1e-9 or b.std() < 1e-9:
            values[i] = 0.0
            continue
        corr = np.corrcoef(a, b)[0, 1]
        values[i] = 0.0 if np.isnan(corr) else corr
    return values


def rolling_xcorr_flags(series_a: np.ndarray, series_b: np.ndarray, window_starts: np.ndarray,
                         window_size: int, corr_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Flags a window where the two series (e.g. gyro rate vs. orientation-channel
    derivative) fail to correlate the way they normally do — axis_decouple breaks
    the physical relationship between related axes without necessarily changing
    either axis's own amplitude, which is why single-channel detectors miss it.
    Returns (window_flags, window_correlations) — the latter is what overlay plots draw.
    """
    values = rolling_xcorr_values(series_a, series_b, window_starts, window_size)
    flags = np.abs(values) < corr_threshold
    return flags, values


def axis_pair_for_channels(channels: list[str]) -> tuple[str, str, bool] | None:
    """
    Pick a physically-related axis pair to cross-correlate, if the device reports one.
    Returns (series_a_col, series_b_col, differentiate_b) — gyro is an angular RATE,
    so it must be compared against the DERIVATIVE of an orientation angle (yaw/pitch/
    roll), not the angle itself, or the correlation is meaningless even for normal data.
    """
    candidates = [
        ("gyro_z", "yaw", True), ("gyro_y", "pitch", True), ("gyro_x", "roll", True),
        ("accel_x", "accel_y", False),
    ]
    for a, b, needs_diff in candidates:
        if a in channels and b in channels:
            return a, b, needs_diff
    return None
