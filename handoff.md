# Handoff: IMU anomaly detection pipeline (thesismorestuff)

Last updated 2026-09-18. Everything below was built and run in one session; nothing has been committed to git.

## Goal

MS thesis: per-device LSTM autoencoder for IMU-based anomaly detection on robotic-arm motion, compared against simple rule-based baselines, per anomaly type (overshoot, undershoot, stall, jitter, drift, early_stop, axis_decouple). Robot joint data is validation/context only, never a model input.

## What exists

| File | Purpose |
|---|---|
| `trim_and_compare.py` | Original task. Trims robot/IMU to the active window (threshold 1 deg, 5-sample pad), clock-overlap cross-check, before/after plots, auto-discovers sessions in `data/`. `pipeline_common.py` imports from it. |
| `pipeline_common.py` | Session loading, sampling-rate check, session-level split, z-score normalizer, windowing. Constants at top (window 77 / step 15, excluded devices, split sizes). |
| `models/autoencoder.py` | PyTorch `LSTMAutoencoder`: enc LSTM(64)->LSTM(32), repeat, dec LSTM(32)->LSTM(64)->Linear per timestep. |
| `train_imu_autoencoder.py` | Trains one model per device; threshold = 95th percentile of validation-window error. |
| `generate_anomalies.py` | Injects all 7 types into held-out **test** sessions only, at localized non-overlapping intervals, saving ground-truth `[start_idx, end_idx)`. |
| `rule_baselines.py` | CUSUM (drift), windowed variance (jitter), DTW to one reference window (overshoot/undershoot/stall/early_stop), rolling cross-correlation (axis_decouple). |
| `evaluate.py` | Scores LSTM and baselines on the same injected test sessions; per-type P/R/F1; overlay plots. |

Naming and style follow the older project at `C:\Users\khatt\Desktop\thesiswork\scripts\` (`train_imu_autoencoder.py`, `generate_anomalies.py`, `models/autoencoder.py`). That project is separate and untouched. Differences from it: full continuous sessions instead of fixed segments, all 12 IMU channels instead of yaw/pitch/roll, z-score instead of MinMax, threshold from validation instead of training windows, the 64->32->32->64 funnel instead of same-size stacked LSTM, and anomalies localized to intervals instead of labelling a whole segment.

## How to run (in order, from `thesismorestuff/`)

```
python train_imu_autoencoder.py --output_dir ./out/models --epochs 60
python generate_anomalies.py --output_dir ./out/anomaly_test_sets
python evaluate.py --model_dir ./out/models --anomaly_dir ./out/anomaly_test_sets --output_dir ./out/eval
```

- Adding a session: drop its robot and IMU CSVs in `data/`. Filenames must match the regexes in `trim_and_compare.py` (`robot_data_session398`, `robot_session400`, `robot session 402`; IMU equivalents).
- `--include_devices 280` overrides the default exclusion of device 280.
- Training took about 3 min per device at 60 epochs on CPU. If launched in the background, stdout is block-buffered, so the log stays empty until the process exits; check `out/models/` for progress.
- The split is deterministic (`seed + device_id`), so train, generate_anomalies and evaluate agree on which sessions are test. Changing the session set reshuffles the split, so retrain and regenerate together.
- Requires torch, pandas, numpy, scipy, matplotlib. TensorFlow/Keras is not installed, which is why the model is PyTorch rather than the Keras layout you described.

## Data facts to know

- **6 sessions found, not 7:** 398, 400, 402, 403, 404, 405. Session 401 does not exist.
- IMU device_ids are 279-283 in every session. Device **271 does not exist** (you confirmed it should not; it was removed). `KNOWN_BASELINE_OFFSETS` in `trim_and_compare.py` is empty.
- Robot `recorded_at` is Unix seconds (float); IMU `recorded_at` is Unix milliseconds (int). `_to_datetime_auto` detects the unit by magnitude.
- One IMU CSV holds several devices interleaved, so always filter by `device_id`.
- Real joint-to-device mapping (`JOINT_DEVICE_MAP` in `trim_and_compare.py`): joint_1 <-> dev 280 yaw; joint_2 <-> dev 281 yaw+pitch; joint_3 <-> dev 282 yaw+pitch **inverted**; joint_4 <-> dev 279 pitch. joint_5 and joint_6 show no motion in session 400 and were dropped from plots.
- Device 280 is excluded by default (`EXCLUDED_DEVICES_DEFAULT`), so the pipeline trains devices 279, 281, 282, 283.

## Verified sampling rates do not match the 100 Hz assumption

Median rate over each trimmed session, from `recorded_at` deltas (`out/models/training_summary.csv` and the startup printout):

| Device | Median Hz | Sessions flagged (>15% off 100 Hz) |
|---|---|---|
| 279 | 71 | 5 of 6 |
| 281 | 56 | 5 of 6 |
| 282 | 91 | 2 of 6 |
| 283 | 53 | 6 of 6 |

18 of 24 device/session combinations are flagged. Two consequences:
1. IMU timestamps are integer milliseconds, so the median dt is quantized (0.011, 0.014, 0.019 s) and the Hz figure is coarse. Check the raw capture rate before trusting the exact numbers.
2. A fixed 77-sample window spans about 0.77 s at 100 Hz but about 1.4 s at 53 Hz. Mixed rates weaken the "same real-time window" assumption. Either confirm the real rate or resample to a common rate before scaling to ~100 sessions.

## Results (6 sessions, 4 devices, 8 held-out test sessions with injected anomalies)

Split per device is 3 train / 1 val / 2 test sessions. Per-type F1, aggregated across test sessions (`out/eval/comparison_table.csv`):

| Type | LSTM F1 | Baseline | Baseline F1 |
|---|---|---|---|
| drift | 0.93 | CUSUM | 0.00 |
| jitter | 0.92 | windowed variance | 0.92 (tie) |
| axis_decouple | 0.93 | rolling xcorr | 0.38 |
| undershoot | 0.60 | DTW | 0.00 |
| overshoot | 0.53 | DTW | 0.00 |
| stall | 0.27 | DTW | 0.00 |
| early_stop | 0.00 | DTW | 0.00 |

Per-device mean LSTM F1 is in `out/eval/device_summary.csv` (279: 0.41, 281: 0.70, 282: 0.80, 283: 0.48). Overlay plots per type are `out/eval/overlay_<type>.png`.

## Caveats and things I would not over-claim

- **Anomalies are synthetic** (`generate_anomalies.py`). All P/R/F1 numbers measure detection of these injections, not real faults. The injection magnitudes (scaled to each channel's std) were chosen by me, not tuned against real failure data.
- **Very small sample:** 8 test sessions, one seed, one split. Validation sets are 97-193 windows, so the 95th-percentile threshold is noisy. No confidence intervals or repeated splits yet.
- **Drift injection was changed mid-session:** it originally left a permanent offset after its interval, which contaminated every later injected type. It is now a triangular ramp that returns to baseline within its interval. That makes it less physically realistic than un-recovered drift.
- **Baseline failures are real limitations, not confirmed-fair tuning.** CUSUM: normal large joint sweeps produce a bigger CUSUM statistic (up to ~355) than the injected drift (max ~93), so no threshold separates them on raw yaw (see `overlay_drift.png`). DTW: one nearest-to-centroid reference window cannot represent every phase of a non-repetitive motion. Stronger baselines (multiple DTW templates, CUSUM on a detrended or model-residual signal) were not tried, so "baseline F1 = 0" partly reflects how simple these baselines are.
- **early_stop LSTM F1 = 0 is at least partly a labeling edge effect.** Reconstruction error rises late in the injected interval, and the crossing window overlaps the interval by less than `OVERLAP_THRESHOLD` (0.5 in `evaluate.py`), so it is excluded from both positives and negatives. Windows partially overlapping any interval are excluded from scoring. Changing the overlap rule or window size will move these numbers.
- Per-type evaluation is a type-vs-normal binary task using the LSTM's single anomaly flag. The LSTM does not classify type.
- `early_stop` was not given its own baseline in your spec; I routed it to the DTW baseline.
- **Robot ground truth is only used as plot context** (bottom panel of overlays). Your brief also described using it to check timing of real motion; no quantitative check of IMU-vs-robot timing is implemented in this pipeline.
- Threshold-based IMU trimming uses yaw/pitch/roll (union of windows) at 1 deg, 5-sample padding. It was not re-validated per device outside session 400.
- `evaluate.py` calibrates CUSUM's threshold (99th percentile x 1.2) and DTW's threshold (mean + 3 std) from training data only; jitter's baseline level is the per-session median window energy.

## Suggested next steps

1. Resolve the sampling-rate question (raw capture rate vs. reported), then decide whether to resample to a common rate.
2. Re-run with several seeds/splits once more sessions exist; report variability.
3. Revisit `OVERLAP_THRESHOLD` and window size (try 30-40 samples for jitter/early_stop) and see how early_stop and stall move.
4. Strengthen baselines if the thesis argument depends on "baseline fails" (multi-template DTW, detrended CUSUM), otherwise state the limitation explicitly.
5. Decide whether device 280 should be re-included (`--include_devices 280`) once its amplitude mismatch is understood.
6. Add a quantitative robot-vs-IMU timing check if you still want that validation.
7. Commit the work; `git status` earlier showed the repo root is the home directory with many unrelated untracked files, so stage only the files listed above.

## Output locations

- `out/session<id>/` : `trim_and_compare.py` before/after plots and summary CSVs
- `out/models/` : trained models, normalizers, training curves, `training_summary.csv`
- `out/anomaly_test_sets/` : labeled test CSVs, label JSONs, `manifest.json`
- `out/eval/` : comparison table, device summary, overlay plots
