# IMU Anomaly Detection for Robotic-Arm Motion

Per-device autoencoders (dense, LSTM, plain RNN) that detect anomalies in IMU recordings of robotic-arm motion, compared against simple rule-based baselines. Built for an MS thesis.

Detection is evaluated per anomaly type: `overshoot`, `undershoot`, `stall`, `jitter`, `drift`, `early_stop`, `axis_decouple`. Robot joint data is used for validation and plot context only; it is never a model input.

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Running the pipeline](#running-the-pipeline)
- [Data](#data)
- [Results](#results)
- [Caveats](#caveats)
- [Development history](#development-history)
- [Next steps](#next-steps)
- [Output locations](#output-locations)

## How it works

1. **Trim** each session to the active motion window and export the trimmed robot and IMU data.
2. **Split** sessions per device into 2 train / 2 validation / 2 test. The split is deterministic (`seed + device_id`).
3. **Train** one autoencoder per (model, device) on clean training windows: all 12 IMU channels, z-score normalized, window 77 samples, step 15.
4. **Inject** synthetic anomalies of all 7 types into the validation and test sessions at localized, non-overlapping intervals, with ground-truth `[start_idx, end_idx)` labels. Nothing is ever injected into training sessions.
5. **Score** each window by reconstruction error. A window is flagged when its error falls outside the central 95% of the clean validation error (below the 2.5th or above the 97.5th percentile).
6. **Evaluate** the models and the rule-based baselines on the same injected test sessions, reporting per-type precision, recall, F1 and AUROC.

### Why the scoring rule is two-sided

Frozen or stalled signals reconstruct *better* than normal motion, so a rule that only flags "error too high" cannot catch them. Under the original one-sided rule, `early_stop` had a test AUROC of 0.02-0.06, which is worse than chance. The two-sided rule fixes this.

The cost is that normal idle windows with very low error are also flagged. Sessions are trimmed to active motion, so this is rare here, but long pauses would raise false positives. `evaluate.py --score_mode one_sided` restores the old rule.

### Models

All three are PyTorch autoencoders defined in `models/autoencoder.py`, each producing a reconstruction of its input window:

| Name | Architecture |
|---|---|
| `dense_ae` | `DenseAutoencoder`, operating on the flattened window |
| `lstm_ae` | `LSTMAutoencoder`: encoder LSTM -> LSTM, repeat, decoder LSTM -> LSTM -> Linear per timestep |
| `rnn_ae` | `RNNAutoencoder`: the same funnel built from plain `nn.RNN` |

### Rule-based baselines

| Anomaly type | Baseline |
|---|---|
| `drift` | CUSUM |
| `jitter` | Windowed variance |
| `overshoot`, `undershoot`, `stall`, `early_stop` | DTW to one reference window |
| `axis_decouple` | Rolling cross-correlation |

## Repository layout

| File | Purpose |
|---|---|
| `trim_and_compare.py` | Trims robot/IMU data to the active window (threshold 1 deg, 5-sample pad), runs a clock-overlap cross-check, writes before/after plots and trimmed CSVs, and auto-discovers sessions in `data/`. `pipeline_common.py` imports from it. |
| `pipeline_common.py` | Session loading, sampling-rate check, session-level split, z-score normalizer, windowing. Constants are at the top (window 77 / step 15, excluded devices, split sizes). |
| `models/autoencoder.py` | The three autoencoders, plus `DEFAULT_HYPERPARAMS`, `get_hyperparams()` (merges in `models/best_hyperparams.json`), `build_model()` and `load_model_for_device()`. |
| `models/best_hyperparams.json` | Winning config per model from the hyperparameter search. Read automatically by training. |
| `train_imu_autoencoder.py` | `fit_model()` (shared with the tuner) and per-(model, device) training with early stopping, denoising noise and a configurable training-window stride. Saves to `models/trained/<model>/`. |
| `tune_hyperparameters.py` | Parallel random search using validation data only. |
| `scoring.py` | Anomaly scoring rules on window reconstruction error: two-sided (default) and one-sided. |
| `export_reconstructions.py` | Reconstructs validation and test sessions per model and writes reconstruction CSVs, PNGs and metrics. |
| `generate_anomalies.py` | Injects all 7 anomaly types and saves ground-truth intervals. `--split test` (default, the reporting set) or `--split val` (the tuning set, with a different seed). |
| `rule_baselines.py` | The rule-based baselines listed above. |
| `evaluate.py` | Scores every model and baseline on the same injected test sessions; per-type P/R/F1 and overlay plots. `--score_mode two_sided\|one_sided`. |

## Requirements

Python with `torch`, `pandas`, `numpy`, `scipy` and `matplotlib`.

## Running the pipeline

Run from the repository root, in this order:

```bash
python trim_and_compare.py                # per-session plots + trimmed CSVs in out/session<id>/
python generate_anomalies.py              # test injections (only if the session set changes)
python generate_anomalies.py --split val  # validation injections (only for tuning)
python tune_hyperparameters.py            # ~1 h on 20 CPUs; writes models/best_hyperparams.json
python train_imu_autoencoder.py           # all 3 models -> models/trained/<model>/ (uses the tuned config)
python export_reconstructions.py          # reconstructions + metrics per model
python evaluate.py                        # models vs rule-based baselines -> out/eval/
```

Useful options:

- `--models lstm_ae rnn_ae` limits which models run.
- `train_imu_autoencoder.py --defaults` ignores the tuned config.
- `--include_devices 280` overrides the default exclusion of device 280.
- `evaluate.py --score_mode one_sided` uses the old one-sided scoring rule.

Notes:

- **Adding a session:** drop its robot and IMU CSVs in `data/`. Filenames must match the regexes in `trim_and_compare.py` (`robot_data_session398`, `robot_session400`, `robot session 402`, and the IMU equivalents).
- **Changing the session set reshuffles the split**, so retrain and regenerate the anomaly sets together. With an unchanged session set, training, `generate_anomalies.py` and `evaluate.py` all agree on which sessions are test.
- Training takes about 3 minutes per device at 60 epochs on CPU. When launched in the background, stdout is block-buffered and the log stays empty until the process exits; check `models/trained/<model>/` for progress.
- Trained model artifacts (`models/trained/`, `models/trained_defaults/`, `models/logs/`) are gitignored and must be regenerated with the scripts.

## Data

- **Sessions:** 398, 400, 402, 403, 404, 405 (six in total; there is no session 401).
- **IMU devices:** IDs 279-283 in every session. Device 280 is excluded by default (`EXCLUDED_DEVICES_DEFAULT`), so the pipeline trains devices 279, 281, 282 and 283.
- **Timestamps:** robot `recorded_at` is Unix seconds (float); IMU `recorded_at` is Unix milliseconds (int). `_to_datetime_auto` detects the unit by magnitude.
- **One IMU CSV holds several devices interleaved**, so always filter by `device_id`.
- **Joint-to-device mapping** (`JOINT_DEVICE_MAP` in `trim_and_compare.py`):

  | Joint | Device | Channels |
  |---|---|---|
  | joint_1 | 280 | yaw |
  | joint_2 | 281 | yaw + pitch |
  | joint_3 | 282 | yaw + pitch, **inverted** |
  | joint_4 | 279 | pitch |

  joint_5 and joint_6 show no motion in session 400 and are dropped from plots.
- **Trimmed exports:** `trimmed_session<id>.csv` holds robot joints plus IMU at the robot rate (the data behind the "after" plot); `trimmed_imu_session<id>.csv` holds the IMU at its native rate. Device 282 yaw and pitch are sign-flipped in these files (the joint-3 inversion is applied before export).
- `KNOWN_BASELINE_OFFSETS` in `trim_and_compare.py` is empty.

### Sampling rates do not match the 100 Hz assumption

Median rate over each trimmed session, computed from `recorded_at` deltas (printed at startup by `train_imu_autoencoder.py`):

| Device | Median Hz | Sessions flagged (>15% off 100 Hz) |
|---|---|---|
| 279 | 71 | 5 of 6 |
| 281 | 56 | 5 of 6 |
| 282 | 91 | 2 of 6 |
| 283 | 53 | 6 of 6 |

18 of 24 device/session combinations are flagged. Two consequences:

1. IMU timestamps are integer milliseconds, so the median dt is quantized (0.011, 0.014, 0.019 s) and the Hz figure is coarse. Check the raw capture rate before trusting the exact numbers.
2. A fixed 77-sample window spans about 0.77 s at 100 Hz but about 1.4 s at 53 Hz. Mixed rates weaken the "same real-time window" assumption. Either confirm the real rate or resample to a common rate before scaling to more sessions.

## Results

6 sessions, 4 devices, 8 held-out test sessions, two-sided scoring, tuned models. All numbers are on untouched test sessions with synthetic injections (`models/trained/summary_metrics.csv`, `out/eval/comparison_table.csv`).

Per-type F1 at the validation-calibrated rule:

| Type | dense_ae | lstm_ae | rnn_ae | Baseline | Baseline F1 |
|---|---|---|---|---|---|
| drift | 0.96 | 0.96 | 0.95 | CUSUM | 0.00 |
| jitter | 0.96 | 0.96 | 0.94 | windowed variance | 0.92 |
| axis_decouple | 0.96 | 0.96 | 0.94 | rolling xcorr | 0.38 |
| undershoot | 0.91 | 0.83 | 0.85 | DTW | 0.00 |
| overshoot | 0.76 | 0.62 | 0.52 | DTW | 0.00 |
| early_stop | 0.92 | 0.88 | 0.87 | DTW | 0.00 |
| stall | 0.52 | 0.60 | 0.58 | DTW | 0.00 |

Model-level, mean over devices:

| Metric | dense_ae | lstm_ae | rnn_ae |
|---|---|---|---|
| Mean per-type test AUROC | 0.925 | 0.927 | 0.930 |
| Mean per-type F1 | 0.83 | 0.82 | 0.80 |
| Reconstruction MSE, clean test (z-score units) | 0.134 | 0.175 | 0.201 |
| Reconstruction R2, clean test | 0.865 | 0.824 | 0.799 |

The models are statistically indistinguishable on detection; `dense_ae` reconstructs best.

What moved the numbers (mean per-type test AUROC):

- **One-sided -> two-sided scoring** took every model from about 0.72-0.74 to about 0.92-0.93. This is the big gain, and almost all of it comes from `early_stop` and `stall`.
- **Hyperparameter tuning** on top of the defaults added only about +0.005 AUROC (F1 +0.02 to +0.06, single seed, so within noise). The default configs were already close to the optimum for this data.

### Tuned hyperparameters

Stored in `models/best_hyperparams.json`. Final training used up to 100 epochs with patience 15.

| Model | Size | Dropout | LR | Weight decay | Batch | Input noise | Train stride |
|---|---|---|---|---|---|---|---|
| `dense_ae` | hidden 64, latent 32 | 0 | 3.2e-4 | 1.4e-4 | 16 | 0.2 | 5 |
| `lstm_ae` | 64 -> 8 -> 8 -> 64 | n/a | 2.3e-3 | ~0 | 16 | 0.05 | 5 |
| `rnn_ae` | 96 -> 16 -> 16 -> 96 | 0.3 | 1.4e-3 | 7.5e-4 | 16 | 0.3 | 10 |

## Caveats

- **Anomalies are synthetic** (`generate_anomalies.py`). All P/R/F1 numbers measure detection of these injections, not real faults. The injection magnitudes (scaled to each channel's std) were chosen by hand, not tuned against real failure data.
- **Very small sample:** 8 test sessions, one seed, one split. Validation sets are 97-193 windows, so the percentile thresholds are noisy. There are no confidence intervals or repeated splits yet.
- **The two-sided gain leans on how the synthetic anomalies look.** `early_stop` and `stall` are injected as flat or frozen segments, which is exactly what the low-error tail catches. Real stalls that keep some motion or noise may be harder.
- **Validation numbers are optimistic.** Tuning and the choice of scoring rule both used the synthetic validation anomalies, so `val_injected` results should not be quoted. Report `test` and `test_injected`, bearing in mind both are still synthetic.
- **`stall` (F1 0.5-0.6) and `overshoot` (0.5-0.8) remain the weakest types**, and `lstm_ae` / `rnn_ae` trail `dense_ae` on overshoot.
- **Baseline failures partly reflect how simple the baselines are.**
  - CUSUM: normal large joint sweeps produce a bigger CUSUM statistic (up to ~355) than the injected drift (max ~93), so no threshold separates them on raw yaw (see `overlay_drift.png`).
  - DTW: one nearest-to-centroid reference window cannot represent every phase of a non-repetitive motion.
  - Stronger baselines (multiple DTW templates, CUSUM on a detrended or model-residual signal) have not been tried.
- **Drift injection is a triangular ramp** that returns to baseline within its interval. An earlier version left a permanent offset that contaminated every later injected type. The current form is less physically realistic than un-recovered drift.
- **Overlap rule:** windows that partially overlap any injected interval are excluded from scoring (`OVERLAP_THRESHOLD` = 0.5). This affects all numbers slightly.
- **Per-type evaluation is a type-vs-normal binary task** using each model's single anomaly flag. The models do not classify anomaly type.
- **`early_stop` has no dedicated baseline**; it is routed to the DTW baseline.
- **Robot ground truth is only used as plot context** (the bottom panel of the overlays). No quantitative check of IMU-vs-robot timing is implemented.
- **Trimming** uses yaw/pitch/roll (union of windows) at 1 deg with 5-sample padding. It has not been re-validated per device outside session 400.
- **Baseline calibration:** `evaluate.py` calibrates CUSUM's threshold (99th percentile x 1.2) and DTW's threshold (mean + 3 std) from training data only; the jitter baseline level is the per-session median window energy.
- **Search budget:** almost every search trial ran the full 60-epoch cap without early stopping (validation loss was still falling), so the search may favor high learning rates.

## Development history

Design decisions and the numbers behind them, in the order they were made (2026-09-18).

### Relationship to the earlier project

Naming and style follow an older, separate thesis project (`train_imu_autoencoder.py`, `generate_anomalies.py`, `models/autoencoder.py`). Differences from it:

- Full continuous sessions instead of fixed segments
- All 12 IMU channels instead of yaw/pitch/roll
- Z-score instead of MinMax normalization
- Thresholds from validation windows instead of training windows
- A 64 -> 32 -> 32 -> 64 funnel instead of same-size stacked LSTMs
- Anomalies localized to intervals instead of labelling a whole segment
- PyTorch instead of Keras

### Restructure, three models, trimmed CSVs

1. Trained artifacts moved from `out/models/` to `models/trained/<model>/`; model code stays in `models/*.py`.
2. `dense_ae` and `rnn_ae` were added next to the existing LSTM autoencoder, each with its own hyperparameters, and training and evaluation were generalized to loop over models. Checkpoints store `model_name` and `arch`, so one loader handles all three.
3. The split became 2 train / 2 val / 2 test per device (`N_VAL_SESSIONS_PER_DEVICE = 2`). The shuffle is seeded and the test sessions are the first two of the shuffle, so the test sessions and the existing anomaly test sets stayed valid (checked for device 279: sessions 398 and 405).
4. `trim_and_compare.py` gained the trimmed CSV exports. The first version had 41 mostly-ID columns and only robot-rate rows (about 17 Hz, because each robot row is matched to its nearest IMU sample), so the columns were cut down and the native-rate IMU file was added.

### Hyperparameter search and the scoring-rule change

5. Training was refactored into a reusable `fit_model()`. Dropout was added to all models, along with Gaussian input noise (denoising) and a separate training-window stride (`train_step`) so that more overlapping training windows can be drawn from only 2 sessions. Validation and test windows always use stride 15, keeping thresholds and labels comparable.
6. A validation-only tuning setup was built: `generate_anomalies.py --split val` uses different sessions and a different seed from the test set, and the tuner scores each config by mean per-type AUROC on those injections. The search is a random search of 60 / 40 / 40 trials for dense / lstm / rnn, across all 4 devices, with one shared config per model and 10 parallel workers. Trial 0 is always the untuned default. The search space covers learning rate, weight decay, batch size, dropout, input noise, training-window stride and network size.
7. **First search (one-sided error as the score).** Validation AUROC, default -> best: dense 0.720 -> 0.765, lstm 0.771 -> 0.775, rnn 0.765 -> 0.780. On the test sessions the winners were no better than the untuned defaults (mean per-type AUROC: dense 0.713 vs 0.715, lstm 0.748 vs 0.742, rnn 0.734 vs 0.731).
8. **Diagnosis.** Per-type AUROC showed `early_stop` at 0.02-0.06 and `stall` at 0.4-0.55 for every model. Frozen or stalled signals reconstruct with *lower* error than normal motion, so an "error too high" rule structurally cannot flag them, however well tuned. An earlier hypothesis, that `early_stop` F1 = 0 was a labeling edge effect, was wrong.
9. **Two-sided scoring** was probed on the validation injections and then confirmed on test. Mean per-type AUROC, one-sided -> two-sided:

   | | dense_ae | lstm_ae | rnn_ae |
   |---|---|---|---|
   | Validation | 0.756 -> 0.897 | 0.777 -> 0.910 | 0.776 -> 0.888 |
   | Test | 0.713 -> 0.914 | 0.748 -> 0.925 | 0.734 -> 0.913 |

   `early_stop` went from about 0.2 to about 0.9 on validation; drift, jitter and axis_decouple lost about 0.01; overshoot lost about 0.05 on validation. Disclosure: the first probe printout accidentally showed the test summary before the validation rows because of an output-filter bug. It was re-run unfiltered, the validation numbers agreed, and the decision rests on those.
10. Under two-sided scoring the first search's winners were *worse* than the untuned defaults on validation (defaults 0.913 / 0.918 / 0.906 vs winners 0.897 / 0.910 / 0.888), so that search had optimized the wrong objective. It is archived in `models/tuning_onesided_objective/`. The search was re-run with two-sided AUROC as the objective. Validation AUROC, default -> best: dense 0.918 -> 0.930, lstm 0.900 -> 0.917, rnn 0.897 -> 0.913.
11. The untuned defaults were also trained into `models/trained_defaults/` to check the effect of tuning on test. Mean per-type test AUROC, tuned vs default: dense 0.925 vs 0.920, lstm 0.927 vs 0.919, rnn 0.930 vs 0.921. Mean per-type F1: 0.833 / 0.819 / 0.802 vs 0.794 / 0.796 / 0.743. Tuning helps slightly and is within single-seed noise.
12. `export_reconstructions.py` was written. Full-session reconstruction is the mean over all windows covering each sample, with a CSV and PNG per session plus reconstruction and detection metrics. `evaluate.py` now uses the two-sided rule by default and writes `threshold_upper` / `threshold_lower` in `device_summary.csv`.

### Known quirk

`models/trained/training_summary.csv` was rebuilt from the per-device metadata files after three parallel training processes overwrote one another. Its columns differ slightly from a single-process run (`threshold_p95`, no sampling-rate columns).

## Next steps

1. Retrain and re-tune when more sessions arrive. With only 2 training sessions per device the tuned gains are within noise, and more data should help more than more tuning. Consider training the final models on train + val once the config is fixed.
2. Resolve the sampling-rate question (raw capture rate vs. reported), then decide whether to resample to a common rate.
3. Re-run with several seeds and splits once more sessions exist, and report variability.
4. Revisit `OVERLAP_THRESHOLD` and window size (try 30-40 samples for jitter and early_stop) and see how `early_stop` and `stall` move.
5. Strengthen the baselines if the thesis argument depends on "baseline fails" (multi-template DTW, detrended CUSUM); otherwise state the limitation explicitly.
6. Decide whether device 280 should be re-included (`--include_devices 280`) once its amplitude mismatch is understood.
7. Add a quantitative robot-vs-IMU timing check.
8. Decide whether to untrack the generated files that are still in git from earlier commits (`git rm -r --cached __pycache__ out`); both are listed in `.gitignore`.

## Output locations

| Path | Contents |
|---|---|
| `out/session<id>/` | Before/after plots, summary CSV, `trimmed_session<id>.csv`, `trimmed_imu_session<id>.csv` |
| `out/anomaly_test_sets/` | Reporting set: labeled CSVs, label JSONs, `manifest.json` |
| `out/anomaly_val_sets/` | Tuning-only set, same layout |
| `out/eval/` | Comparison table (models vs baselines), device summary, overlay plots |
| `models/trained/<model>/` | Model, normalizer and training curve per device; `reconstruction/{val,test,test_injected,val_injected}/`; `metrics_reconstruction.csv`; `metrics_detection.csv` |
| `models/trained/summary_metrics.csv` | Metrics across all models |
| `models/trained/training_summary.csv` | Training summary per (model, device) |
| `models/trained_defaults/` | Same layout for the untuned-default models |
| `models/best_hyperparams.json` | Tuned config per model |
| `models/tuning/` | Results of the final (two-sided) search |
| `models/tuning_onesided_objective/` | The discarded first search |
| `models/logs/` | Training, export and evaluation logs |
