"""
Anomaly scoring rules on top of window reconstruction error
===========================================================
One-sided rule (the usual autoencoder rule): flag a window when its reconstruction error is
ABOVE the 95th percentile of clean validation error.

Two-sided rule (default in this pipeline): flag a window when its error is outside the central
95% of clean validation error, i.e. below the 2.5th OR above the 97.5th percentile.

Why two-sided: an autoencoder reconstructs simple signals more easily than complex ones. Normal
active motion has moderate error; a frozen / stalled / early-stopped signal is nearly constant
and reconstructs with LOWER error than normal, so a one-sided rule scores those anomalies as
"more normal than normal" (AUROC well below 0.5). The two-sided rule needs no anomaly data to
calibrate: both bounds come from clean validation windows only, and the total false-positive
rate on clean validation stays at 5%.

Trade-off: a genuinely idle-but-normal window (very low error) is now flagged too. Sessions are
trimmed to the active motion window, which keeps such windows rare, but very long normal pauses
would raise false positives.
"""

from __future__ import annotations

import numpy as np
import torch

CENTRAL_FRACTION = 0.95  # share of clean validation error treated as normal


def val_reference_errors(model, normalizer, val_sessions, channels: list[str],
                         window_size: int, window_step: int) -> np.ndarray:
    """Window reconstruction errors on the device's clean validation sessions (the calibration set)."""
    from pipeline_common import make_windows

    errs = []
    for s in val_sessions:
        arr = normalizer.transform(s.df[channels].values.astype(np.float64))
        windows, _ = make_windows(arr, window_size, window_step)
        if len(windows):
            with torch.no_grad():
                errs.append(model.reconstruction_error(torch.tensor(windows, dtype=torch.float32)).cpu().numpy())
    return np.concatenate(errs) if errs else np.array([])


def two_sided_bounds(ref_errors: np.ndarray, central: float = CENTRAL_FRACTION) -> tuple[float, float]:
    tail = (1 - central) / 2 * 100
    return float(np.percentile(ref_errors, tail)), float(np.percentile(ref_errors, 100 - tail))


def two_sided_flags(errors: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return (errors < lo) | (errors > hi)


def two_sided_score(errors: np.ndarray, ref_errors: np.ndarray) -> np.ndarray:
    """Threshold-free anomaly score: distance of the error's validation-CDF rank from the median (0..0.5)."""
    ranks = np.searchsorted(np.sort(ref_errors), errors) / len(ref_errors)
    return np.abs(ranks - 0.5)
