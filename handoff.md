# Handoff: IMU anomaly detection pipeline (thesismorestuff)

Last updated 2026-09-18. Everything below was built and run over two working sessions on the same day; nothing has been committed to git. The "Work log" section records what was done and in what order, including what was tried and discarded.

## Goal

MS thesis: per-device autoencoders (dense, LSTM, plain RNN) for IMU-based anomaly detection on robotic-arm motion, compared against simple rule-based baselines, per anomaly type (overshoot, undershoot, stall, jitter, drift, early_stop, axis_decouple). Robot joint data is validation/context only, never a model input.

## What exists

| File | Purpose |
|---|---|
| `trim_and_compare.py` | Original task. Trims robot/IMU to the active window (threshold 1 deg, 5-sample pad), clock-overlap cross-check, before/after plots, auto-discovers sessions in `data/`. `pipeline_common.py` imports from it. |
| `pipeline_common.py` | Session loading, sampling-rate check, session-level split, z-score normalizer, windowing. Constants at top (window 77 / step 15, excluded devices, split sizes). |
| `models/autoencoder.py` | Three PyTorch autoencoders that each output a reconstruction: `DenseAutoencoder` (flattened window), `LSTMAutoencoder` (enc LSTM->LSTM, repeat, dec LSTM->LSTM->Linear per timestep) and `RNNAutoencoder` (same funnel, plain `nn.RNN`). Also `DEFAULT_HYPERPARAMS`, `get_hyperparams()` (merges in `models/best_hyperparams.json`), `build_model()`, `load_model_for_device()`. |
| `models/best_hyperparams.json` | Winning config per model from the search. Read automatically by training. |
| `train_imu_autoencoder.py` | `fit_model()` (shared with the tuner) + per-(model, device) training with early stopping, denoising noise, configurable training-window stride. Saves to `models/trained/<model>/`. Keeps a one-sided 95th-percentile threshold in the metadata. |
| `tune_hyperparameters.py` | Parallel random search on validation data only (see the work log). |
| `scoring.py` | Anomaly scoring rules on window reconstruction error: two-sided (default) and one-sided. |
| `export_reconstructions.py` | Reconstructs val/test sessions per model and writes reconstruction CSVs/PNGs and metrics. |
| `generate_anomalies.py` | Injects all 7 types at localized non-overlapping intervals, saving ground-truth `[start_idx, end_idx)`. `--split test` (default, reporting set) or `--split val` (tuning set, different seed). Never into train sessions. |
| `rule_baselines.py` | CUSUM (drift), windowed variance (jitter), DTW to one reference window (overshoot/undershoot/stall/early_stop), rolling cross-correlation (axis_decouple). |
| `evaluate.py` | Scores every model and the baselines on the same injected test sessions; per-type P/R/F1; overlay plots. `--score_mode two_sided|one_sided`. |

Naming and style follow the older project at `C:\Users\khatt\Desktop\thesiswork\scripts\` (`train_imu_autoencoder.py`, `generate_anomalies.py`, `models/autoencoder.py`). That project is separate and untouched. Differences from it: full continuous sessions instead of fixed segments, all 12 IMU channels instead of yaw/pitch/roll, z-score instead of MinMax, threshold from validation instead of training windows, the 64->32->32->64 funnel instead of same-size stacked LSTM, and anomalies localized to intervals instead of labelling a whole segment.

## Summary of changes on 2026-09-18 (models, tuning, scoring, reconstructions)

- **Three models**: `dense_ae`, `lstm_ae`, `rnn_ae` in `models/autoencoder.py`, each with its own hyperparameters (`DEFAULT_HYPERPARAMS`, overridden by the tuned `models/best_hyperparams.json`). Trained artifacts live in `models/trained/<model>/` (gitignored), not `out/`.
- **Split is 2 train / 2 val / 2 test per device.** Test sessions did not change, so `out/anomaly_test_sets/` is still valid.
- **Hyperparameter search** (`tune_hyperparameters.py`): random search, 60 trials dense_ae / 40 lstm_ae / 40 rnn_ae, all 4 devices each, one shared config per model. Selection uses only validation data: val sessions with anomalies injected under a different seed (`python generate_anomalies.py --split val` -> `out/anomaly_val_sets/`), scored by mean per-type AUROC. Search space: lr, weight decay, batch size, dropout, input-noise (denoising), training-window stride, network size. Trial 0 is always the untuned default. Logs and every trial: `models/tuning/`.
- **Scoring rule changed to two-sided** (`scoring.py`). A window is flagged when its reconstruction error is outside the central 95% of clean validation error (below the 2.5th OR above the 97.5th percentile). Reason: frozen/stalled signals reconstruct *better* than normal motion, so the old one-sided rule gave `early_stop` a test AUROC of 0.02-0.06 (worse than chance). Adopted on validation evidence, then confirmed on test. Cost: normal idle windows with very low error would now be flagged; sessions are trimmed to active motion so this is rare here, but long pauses would raise false positives. `evaluate.py --score_mode one_sided` restores the old rule. The first search (`models/tuning_onesided_objective/`) optimized the one-sided score and was discarded because its winners were *worse* than the defaults under the two-sided score.
- **Reconstructions + metrics**: `python export_reconstructions.py` writes, per model, `reconstruction/{val,test,test_injected,val_injected}/device<D>_session<S>.{csv,png}` (original vs reconstructed signal in original units, window error, flags) plus `metrics_reconstruction.csv`, `metrics_detection.csv`, and `models/trained/summary_metrics.csv` across models. `val_injected` drove hyperparameter selection, so its numbers are optimistic; report `test` and `test_injected`.
- **Trimmed-movement CSVs**: `trim_and_compare.py` writes `out/session<id>/trimmed_session<id>.csv` (robot joints + IMU at the robot rate; data behind the "after" plot) and `trimmed_imu_session<id>.csv` (native IMU rate). Device 282 yaw/pitch are sign-flipped (joint-3 inversion applied before export).
- `models/trained_defaults/` holds the untuned-default models, kept so the effect of tuning can be checked on test.

## Work log (in order, with the numbers behind each decision)

**Request 1: restructure, three models, trimmed CSVs**
1. Moved trained artifacts from `out/models/` to `models/trained/<model>/` (code stays in `models/*.py`). Added `models/trained/` to `.gitignore`.
2. Added `dense_ae` and `rnn_ae` next to the existing LSTM autoencoder, with per-model hyperparameters, and generalized training/evaluation to loop over models. Model checkpoints now store `model_name` and `arch`, so one loader handles all three. The old single-LSTM checkpoints were incompatible and were superseded.
3. Changed the split to 2 train / 2 val / 2 test per device (`N_VAL_SESSIONS_PER_DEVICE = 2`). The shuffle is seeded and the test sessions are the first two of the shuffle, so the test sessions are identical to before and the existing anomaly test sets stayed valid (checked for device 279: 398 and 405).
4. `trim_and_compare.py` now exports `trimmed_session<id>.csv` (the data behind the "after" plot) and `trimmed_imu_session<id>.csv`. First version had 41 mostly-ID columns and only robot-rate rows (about 17 Hz, because each robot row is matched to its nearest IMU sample), so I trimmed the columns and added the native-rate IMU file.
5. Smoke-tested everything with 2-epoch models in a scratch folder before any real run.

**Request 2: rigorous hyperparameter search, reconstructions on val and test, metrics, accuracy improvements**
6. Refactored training into a reusable `fit_model()`; added dropout to all models, Gaussian input noise (denoising), and a separate training-window stride (`train_step`) so more overlapping training windows can be drawn from only 2 sessions. Val/test windows always use stride 15 so thresholds and labels stay comparable.
7. Built a validation-only tuning setup: added `generate_anomalies.py --split val` (different sessions and seed from the test set), and the tuner scores each config by mean per-type AUROC on those val injections. Benchmarked one job first (about 1 s per epoch) and sized the search from that: 60 / 40 / 40 trials for dense / lstm / rnn, 4 devices each, 10 parallel workers, trial 0 = defaults.
8. **First search (one-sided error as the score).** Val AUROC, default -> best: dense 0.720 -> 0.765, lstm 0.771 -> 0.775, rnn 0.765 -> 0.780. I trained the winners for 100 epochs and evaluated: on the test sessions they were no better than the untuned defaults (mean per-type AUROC dense 0.713 vs 0.715, lstm 0.748 vs 0.742, rnn 0.734 vs 0.731).
9. **Diagnosis.** Per-type AUROC showed `early_stop` at 0.02-0.06 and `stall` at 0.4-0.55 for every model: frozen or stalled signals reconstruct with *lower* error than normal motion, so a "error too high" rule structurally cannot flag them, however well tuned. Shape anomalies (overshoot/undershoot) were also mediocre.
10. **Two-sided scoring** (`scoring.py`). Probed on the val injections first, then test: val mean per-type AUROC one-sided -> two-sided: dense 0.756 -> 0.897, lstm 0.777 -> 0.910, rnn 0.776 -> 0.888; test: 0.713 -> 0.914, 0.748 -> 0.925, 0.734 -> 0.913. `early_stop` went from about 0.2 (val) to about 0.9; drift/jitter/axis_decouple lost about 0.01; overshoot lost about 0.05 on val. Adopted as the default rule. Disclosure: the first probe printout accidentally showed the test summary before the val rows (an output-filter bug in my command); I then re-ran unfiltered and looked at val, which agreed, and the decision rests on the val numbers.
11. Under two-sided scoring the first search's winners were *worse* than the untuned defaults on val (defaults 0.913 / 0.918 / 0.906 vs winners 0.897 / 0.910 / 0.888), so that search was optimizing the wrong objective. Archived it in `models/tuning_onesided_objective/` and **re-ran the search with the two-sided AUROC as the objective**. Val AUROC, default -> best: dense 0.918 -> 0.930, lstm 0.900 -> 0.917, rnn 0.897 -> 0.913.
12. Final tuned configs (in `models/best_hyperparams.json`): dense_ae hidden 64, latent 32, dropout 0, lr 3.2e-4, weight decay 1.4e-4, batch 16, noise 0.2, train stride 5. lstm_ae 64->8->8->64, lr 2.3e-3, weight decay about 0, batch 16, noise 0.05, stride 5. rnn_ae 96->16->16->96, dropout 0.3, lr 1.4e-3, weight decay 7.5e-4, batch 16, noise 0.3, stride 10. Final training used up to 100 epochs, patience 15.
13. Trained the untuned defaults into `models/trained_defaults/` too, to check the effect of tuning on test. Mean per-type test AUROC tuned vs default (two-sided): dense 0.925 vs 0.920, lstm 0.927 vs 0.919, rnn 0.930 vs 0.921; mean per-type F1 0.833 / 0.819 / 0.802 vs 0.794 / 0.796 / 0.743. So tuning helps slightly and is within single-seed noise.
14. Wrote `export_reconstructions.py` (full-session reconstruction = mean over all windows covering each sample; CSV and PNG per session; reconstruction and detection metrics) and ran it for the tuned and default models. Updated `evaluate.py` to use the two-sided rule by default and to write `threshold_upper`/`threshold_lower` in `device_summary.csv`.

**Things that went wrong or need knowing**
- Almost every search trial ran the full 60-epoch cap without early stopping (val loss was still falling), so the search budget may favor high learning rates. Final training used up to 100 epochs.
- Three training processes wrote `training_summary.csv` at the same time, so it only held one model; I rebuilt it from the per-device metadata files (columns differ slightly: `threshold_p95`, no sampling-rate columns).
- Before the final training I deleted `models/trained/` (stale, gitignored artifacts from the first search's winners and the parked old single-LSTM checkpoints) so no old file could be mistaken for a current one. The defaults are still in `models/trained_defaults/`.
- Wrong first hypothesis: I had guessed early_stop F1 = 0 was a labeling edge effect. It was the one-sided scoring rule (step 9).
- Git state: the previous commit (`d7a87a8 test`) tracked `out/` and `__pycache__/`, so the working tree shows the old `out/models/*` as deleted and `out/session*` / `out/eval` files as modified. Nothing was committed.

## How to run (in order, from the repo root)

```
python trim_and_compare.py                                   # per-session plots + trimmed CSVs in out/session<id>/
python generate_anomalies.py                                 # test injections   (only if the session set changes)
python generate_anomalies.py --split val                     # val injections    (only for tuning)
python tune_hyperparameters.py                               # ~1 h on 20 CPUs; writes models/best_hyperparams.json
python train_imu_autoencoder.py                              # all 3 models -> models/trained/<model>/ (uses the tuned config)
python export_reconstructions.py                             # reconstructions + metrics per model
python evaluate.py                                           # models vs rule-based baselines -> out/eval/
```

`--models lstm_ae rnn_ae` limits which models run; `train_imu_autoencoder.py --defaults` ignores the tuned config.

- Adding a session: drop its robot and IMU CSVs in `data/`. Filenames must match the regexes in `trim_and_compare.py` (`robot_data_session398`, `robot_session400`, `robot session 402`; IMU equivalents).
- `--include_devices 280` overrides the default exclusion of device 280.
- Training took about 3 min per device at 60 epochs on CPU. If launched in the background, stdout is block-buffered, so the log stays empty until the process exits; check `models/trained/<model>/` for progress.
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

Median rate over each trimmed session, from `recorded_at` deltas (startup printout of `train_imu_autoencoder.py`):

| Device | Median Hz | Sessions flagged (>15% off 100 Hz) |
|---|---|---|
| 279 | 71 | 5 of 6 |
| 281 | 56 | 5 of 6 |
| 282 | 91 | 2 of 6 |
| 283 | 53 | 6 of 6 |

18 of 24 device/session combinations are flagged. Two consequences:
1. IMU timestamps are integer milliseconds, so the median dt is quantized (0.011, 0.014, 0.019 s) and the Hz figure is coarse. Check the raw capture rate before trusting the exact numbers.
2. A fixed 77-sample window spans about 0.77 s at 100 Hz but about 1.4 s at 53 Hz. Mixed rates weaken the "same real-time window" assumption. Either confirm the real rate or resample to a common rate before scaling to ~100 sessions.

## Results (6 sessions, 4 devices, 8 held-out test sessions, two-sided scoring, tuned models)

Untouched test sessions with synthetic injections (`models/trained/summary_metrics.csv`, `out/eval/comparison_table.csv`). Per-type F1 at the validation-calibrated rule:

| Type | dense_ae | lstm_ae | rnn_ae | Baseline | Baseline F1 |
|---|---|---|---|---|---|
| drift | 0.96 | 0.96 | 0.95 | CUSUM | 0.00 |
| jitter | 0.96 | 0.96 | 0.94 | windowed variance | 0.92 |
| axis_decouple | 0.96 | 0.96 | 0.94 | rolling xcorr | 0.38 |
| undershoot | 0.91 | 0.83 | 0.85 | DTW | 0.00 |
| overshoot | 0.76 | 0.62 | 0.52 | DTW | 0.00 |
| early_stop | 0.92 | 0.88 | 0.87 | DTW | 0.00 |
| stall | 0.52 | 0.60 | 0.58 | DTW | 0.00 |

Model-level (mean over devices): test mean per-type AUROC 0.925 / 0.927 / 0.930 (dense / lstm / rnn), mean per-type F1 0.83 / 0.82 / 0.80. Reconstruction on clean test sessions (z-score units): MSE 0.134 / 0.175 / 0.201, R2 0.865 / 0.824 / 0.799. The models are statistically indistinguishable on detection; dense_ae reconstructs best.

What moved the numbers (mean per-type test AUROC): one-sided -> two-sided scoring took every model from ~0.72-0.74 to ~0.92-0.93 (the big gain, almost all from early_stop and stall). Tuning on top of the defaults added only about +0.005 AUROC (F1 +0.02 to +0.06, single seed, so within noise). The default configs were already close to the optimum for this data.

## Caveats and things I would not over-claim

- **Anomalies are synthetic** (`generate_anomalies.py`). All P/R/F1 numbers measure detection of these injections, not real faults. The injection magnitudes (scaled to each channel's std) were chosen by me, not tuned against real failure data.
- **Very small sample:** 8 test sessions, one seed, one split. Validation sets are 97-193 windows, so the 95th-percentile threshold is noisy. No confidence intervals or repeated splits yet.
- **Drift injection was changed mid-session:** it originally left a permanent offset after its interval, which contaminated every later injected type. It is now a triangular ramp that returns to baseline within its interval. That makes it less physically realistic than un-recovered drift.
- **Baseline failures are real limitations, not confirmed-fair tuning.** CUSUM: normal large joint sweeps produce a bigger CUSUM statistic (up to ~355) than the injected drift (max ~93), so no threshold separates them on raw yaw (see `overlay_drift.png`). DTW: one nearest-to-centroid reference window cannot represent every phase of a non-repetitive motion. Stronger baselines (multiple DTW templates, CUSUM on a detrended or model-residual signal) were not tried, so "baseline F1 = 0" partly reflects how simple these baselines are.
- **The earlier early_stop F1 = 0 was NOT a labeling edge effect** (that was the previous hypothesis, now disproven): frozen signals reconstruct with lower error than normal motion, which a one-sided rule cannot flag. The two-sided rule fixes it. The overlap rule (`OVERLAP_THRESHOLD` = 0.5; windows partially overlapping any interval are excluded from scoring) still affects all numbers slightly.
- **The two-sided gain leans on how the synthetic anomalies look.** early_stop and stall are injected as flat/frozen segments, which is exactly what the low-error tail catches. Real stalls that keep some motion or noise may be harder. Also, tuning and scoring-rule selection used synthetic val anomalies, so they are optimistic; the test numbers are the ones to quote, and both are still synthetic.
- **stall (F1 0.5-0.6) and overshoot (0.5-0.8) remain the weakest types**, and lstm/rnn trail dense_ae on overshoot. Stronger baselines were still not tried.
- Per-type evaluation is a type-vs-normal binary task using the LSTM's single anomaly flag. The LSTM does not classify type.
- `early_stop` was not given its own baseline in your spec; I routed it to the DTW baseline.
- **Robot ground truth is only used as plot context** (bottom panel of overlays). Your brief also described using it to check timing of real motion; no quantitative check of IMU-vs-robot timing is implemented in this pipeline.
- Threshold-based IMU trimming uses yaw/pitch/roll (union of windows) at 1 deg, 5-sample padding. It was not re-validated per device outside session 400.
- `evaluate.py` calibrates CUSUM's threshold (99th percentile x 1.2) and DTW's threshold (mean + 3 std) from training data only; jitter's baseline level is the per-session median window energy.

## Suggested next steps

0. Retrain/re-tune when more sessions arrive: with only 2 training sessions per device the tuned gains are within noise; more data should help more than more tuning. Consider training the final models on train+val once the config is fixed.
1. Resolve the sampling-rate question (raw capture rate vs. reported), then decide whether to resample to a common rate.
2. Re-run with several seeds/splits once more sessions exist; report variability.
3. Revisit `OVERLAP_THRESHOLD` and window size (try 30-40 samples for jitter/early_stop) and see how early_stop and stall move.
4. Strengthen baselines if the thesis argument depends on "baseline fails" (multi-template DTW, detrended CUSUM), otherwise state the limitation explicitly.
5. Decide whether device 280 should be re-included (`--include_devices 280`) once its amplitude mismatch is understood.
6. Add a quantitative robot-vs-IMU timing check if you still want that validation.
7. Commit the work. The repo root is `Anomalydetection/` and the previous commit tracked `out/` and `__pycache__/`; decide whether to untrack those (`git rm -r --cached __pycache__ out`) before committing. `models/trained*/` and `models/logs/` are gitignored; `models/best_hyperparams.json` and `models/tuning/` are worth committing.

## Output locations

- `out/session<id>/` : `trim_and_compare.py` before/after plots, summary CSV, `trimmed_session<id>.csv`, `trimmed_imu_session<id>.csv`
- `models/trained/<model_name>/` : model + normalizer + training curve per device, `reconstruction/{val,test,test_injected,val_injected}/`, `metrics_reconstruction.csv`, `metrics_detection.csv`
- `models/trained/summary_metrics.csv` (all models) and `models/trained/training_summary.csv`
- `models/trained_defaults/` : same layout for the untuned-default models
- `models/best_hyperparams.json` : tuned config per model
- `models/tuning/` : search results for the final (two-sided) search; `models/tuning_onesided_objective/` : the discarded first search
- `models/logs/` : training/export/evaluate logs
- `out/anomaly_test_sets/` (reporting) and `out/anomaly_val_sets/` (tuning only) : labeled CSVs, label JSONs, `manifest.json`
- `out/eval/` : comparison table (models vs baselines), device summary, overlay plots
