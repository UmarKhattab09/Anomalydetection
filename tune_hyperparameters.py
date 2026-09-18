"""
IMU Anomaly Detection — Hyperparameter search (random search, validation sessions only)
=======================================================================================
Random search per model (dense_ae, lstm_ae, rnn_ae). Trial 0 is always the current
default config so you can see whether tuning actually helped.

How a config is scored (no test data is ever touched)
-----------------------------------------------------
For each device: fit on that device's TRAIN sessions (early stopping on clean VAL windows),
then score the same device's VAL sessions with synthetic anomalies injected into them
(generate_anomalies.py --split val, a different seed than the test set). The score is the
mean over the 7 anomaly types of the window-level AUROC of the TWO-SIDED anomaly score
(scoring.py: how far a window's reconstruction error sits from the median of clean val error,
in either direction; type windows vs. normal windows). AUROC is threshold-free, so it isn't hurt
by a noisy percentile threshold. The one-sided AUROC (plain error) is logged for reference. A config's score is the mean over devices; one shared config
per model is chosen (per-device tuning on 2 train sessions would just overfit).

Caveat: the val sessions serve both as early-stopping data (clean copy) and as the source
of the injected copy used for selection, and the anomalies are synthetic. Selection is
therefore slightly optimistic, which is why final numbers are reported on the untouched
test sessions.

Usage
-----
  python generate_anomalies.py --split val          # once
  python tune_hyperparameters.py                    # all models, default trial counts
  python tune_hyperparameters.py --models lstm_ae --trials 20 --workers 8

Outputs
-------
  models/tuning/<model>_jobs.jsonl      raw per-(trial, device) results (enables resuming)
  models/tuning/<model>_trials.csv      one row per trial, ranked
  models/best_hyperparams.json          winning config per model (read by train_imu_autoencoder.py)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd

from models.autoencoder import BEST_HYPERPARAMS_PATH, DEFAULT_HYPERPARAMS, MODEL_CLASSES

TUNING_DIR = os.path.join("models", "tuning")
VAL_ANOMALY_DIR = os.path.join("out", "anomaly_val_sets")

# Search-time budget (final training uses the config's epochs/patience below).
SEARCH_EPOCHS = 60
SEARCH_PATIENCE = 10
FINAL_EPOCHS = 100
FINAL_PATIENCE = 15

DEFAULT_TRIALS = {"dense_ae": 60, "lstm_ae": 40, "rnn_ae": 40}


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------

def sample_config(model_name: str, rng: np.random.Generator) -> dict:
    hp = {
        "epochs": SEARCH_EPOCHS, "patience": SEARCH_PATIENCE,
        "batch_size": int(rng.choice([16, 32, 64])),
        "lr": float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
        "weight_decay": float(10 ** rng.uniform(-6, -3)),
        "noise_std": float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.3])),
        "train_step": int(rng.choice([5, 10, 15])),
    }
    dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
    if model_name == "dense_ae":
        hp["arch"] = {"hidden": int(rng.choice([64, 128, 256, 512])),
                      "latent": int(rng.choice([4, 8, 16, 32, 64])), "dropout": dropout}
    else:  # lstm_ae / rnn_ae: mirrored funnel
        h1 = int(rng.choice([16, 32, 48, 64, 96]))
        h2 = int(rng.choice([4, 8, 16, 32]))
        hp["arch"] = {"enc_hidden1": h1, "enc_hidden2": h2, "dec_hidden1": h2, "dec_hidden2": h1,
                      "dropout": dropout}
    return hp


def build_trials(model_name: str, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    default = copy.deepcopy(DEFAULT_HYPERPARAMS[model_name])
    default["epochs"], default["patience"] = SEARCH_EPOCHS, SEARCH_PATIENCE
    trials = [default] + [sample_config(model_name, rng) for _ in range(n_trials - 1)]
    return trials


# --------------------------------------------------------------------------
# Worker (runs in a subprocess)
# --------------------------------------------------------------------------

def _init_worker(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)


@lru_cache(maxsize=1)
def _load_splits():
    from pipeline_common import EXCLUDED_DEVICES_DEFAULT, load_all_device_sessions, session_level_split
    device_sessions = load_all_device_sessions(excluded_devices=EXCLUDED_DEVICES_DEFAULT.copy())
    return session_level_split(device_sessions)


@lru_cache(maxsize=1)
def _load_val_manifest():
    with open(os.path.join(VAL_ANOMALY_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    by_device: dict[int, list[dict]] = {}
    for e in manifest:
        by_device.setdefault(e["device_id"], []).append(e)
    return by_device


def run_job(job: dict) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score

    from evaluate import label_windows
    from generate_anomalies import ANOMALY_TYPES
    from pipeline_common import WINDOW_SIZE, WINDOW_STEP, ZScoreNormalizer, make_windows
    from scoring import two_sided_score
    from train_imu_autoencoder import build_windows_for_sessions, fit_model

    t0 = time.time()
    device_id, hp = job["device_id"], job["hp"]
    split = _load_splits()[device_id]
    channels = split.train[0].channels
    for s in split.train + split.val + split.test:
        channels = [c for c in channels if c in s.channels]

    normalizer = ZScoreNormalizer(channels=channels).fit(split.train)
    train_w = build_windows_for_sessions(split.train, channels, normalizer, WINDOW_SIZE, hp["train_step"])
    val_w = build_windows_for_sessions(split.val, channels, normalizer, WINDOW_SIZE, WINDOW_STEP)

    fit = fit_model(job["model_name"], len(channels), WINDOW_SIZE, train_w, val_w, hp,
                    torch.device("cpu"), seed=job["seed"], verbose=False)
    model = fit["model"]

    # Score val sessions with injected anomalies: per-type AUROC of window reconstruction error.
    with torch.no_grad():
        ref_errors = model.reconstruction_error(torch.tensor(val_w, dtype=torch.float32)).numpy()
    pos_neg: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {t: [] for t in ANOMALY_TYPES}
    for entry in _load_val_manifest().get(device_id, []):
        df = pd.read_csv(entry["csv"])
        with open(entry["labels"]) as f:
            labels = json.load(f)
        arr = normalizer.transform(df[channels].values.astype(np.float64))
        windows, starts = make_windows(arr, WINDOW_SIZE, WINDOW_STEP)
        if not len(windows):
            continue
        with torch.no_grad():
            errors = model.reconstruction_error(torch.tensor(windows, dtype=torch.float32)).numpy()
        gt = label_windows(starts, WINDOW_SIZE, labels)
        for t in ANOMALY_TYPES:
            mask = gt[t] | gt["normal"]
            if gt[t].sum() > 0:
                pos_neg[t].append((gt[t][mask], two_sided_score(errors[mask], ref_errors), errors[mask]))

    per_type, per_type_onesided = {}, {}
    for t, parts in pos_neg.items():
        if not parts:
            continue
        y = np.concatenate([p[0] for p in parts])
        if 0 < y.sum() < len(y):
            per_type[t] = float(roc_auc_score(y, np.concatenate([p[1] for p in parts])))
            per_type_onesided[t] = float(roc_auc_score(y, np.concatenate([p[2] for p in parts])))

    return {
        "model_name": job["model_name"], "trial": job["trial"], "device_id": device_id,
        "hp": hp, "best_val_loss": fit["best_val_loss"], "epochs_run": fit["epochs_run"],
        "val_auroc_by_type": per_type,
        "val_auroc_mean": float(np.mean(list(per_type.values()))) if per_type else float("nan"),
        "val_auroc_onesided_mean": float(np.mean(list(per_type_onesided.values()))) if per_type_onesided else float("nan"),
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _jobs_path(model_name: str) -> str:
    return os.path.join(TUNING_DIR, f"{model_name}_jobs.jsonl")


def _load_done(model_name: str) -> list[dict]:
    path = _jobs_path(model_name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(model_name: str, n_devices: int) -> pd.DataFrame:
    rows = _load_done(model_name)
    by_trial: dict[int, list[dict]] = {}
    for r in rows:
        by_trial.setdefault(r["trial"], []).append(r)
    out = []
    for trial, rs in by_trial.items():
        if len(rs) < n_devices:
            continue  # incomplete trial (interrupted)
        hp = rs[0]["hp"]
        row = {
            "trial": trial,
            "val_auroc_mean": float(np.mean([r["val_auroc_mean"] for r in rs])),
            "val_auroc_onesided_mean": float(np.mean([r.get("val_auroc_onesided_mean", np.nan) for r in rs])),
            "val_loss_mean": float(np.mean([r["best_val_loss"] for r in rs])),
            "epochs_mean": float(np.mean([r["epochs_run"] for r in rs])),
            "minutes": round(sum(r["seconds"] for r in rs) / 60, 1),
            "lr": hp["lr"], "weight_decay": hp["weight_decay"], "batch_size": hp["batch_size"],
            "noise_std": hp["noise_std"], "train_step": hp["train_step"],
        }
        row.update({f"arch_{k}": v for k, v in hp["arch"].items()})
        for t in sorted({t for r in rs for t in r["val_auroc_by_type"]}):
            row[f"auroc_{t}"] = float(np.mean([r["val_auroc_by_type"][t] for r in rs if t in r["val_auroc_by_type"]]))
        out.append(row)
    df = pd.DataFrame(out).sort_values("val_auroc_mean", ascending=False).reset_index(drop=True)
    return df


def tune_model(model_name: str, n_trials: int, workers: int, threads: int, seed: int, device_ids: list[int]) -> None:
    os.makedirs(TUNING_DIR, exist_ok=True)
    trials = build_trials(model_name, n_trials, seed)
    done = {(r["trial"], r["device_id"]) for r in _load_done(model_name)}
    jobs = [{"model_name": model_name, "trial": i, "device_id": d, "hp": hp, "seed": 42}
            for i, hp in enumerate(trials) for d in device_ids if (i, d) not in done]
    print(f"\n[{model_name}] {len(trials)} trials x {len(device_ids)} devices = {len(trials) * len(device_ids)} jobs "
          f"({len(jobs)} to run, {len(done)} already done)", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,)) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with open(_jobs_path(model_name), "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"  [{model_name}] {n}/{len(jobs)} trial {res['trial']} dev {res['device_id']}: "
                  f"auroc={res['val_auroc_mean']:.3f} val_loss={res['best_val_loss']:.3f} "
                  f"ep={res['epochs_run']} ({res['seconds']}s, elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    df = summarize(model_name, len(device_ids))
    df.to_csv(os.path.join(TUNING_DIR, f"{model_name}_trials.csv"), index=False)
    print(f"\n[{model_name}] top 5 trials by mean val AUROC:")
    print(df.head(5).to_string(index=False), flush=True)
    default_row = df[df["trial"] == 0]
    if len(default_row):
        print(f"[{model_name}] defaults (trial 0): val AUROC {default_row['val_auroc_mean'].iloc[0]:.4f}; "
              f"best: {df['val_auroc_mean'].iloc[0]:.4f} (trial {int(df['trial'].iloc[0])})", flush=True)

    best_trial = int(df["trial"].iloc[0])
    best_hp = copy.deepcopy(trials[best_trial])
    best_hp["epochs"], best_hp["patience"] = FINAL_EPOCHS, FINAL_PATIENCE
    best_hp["_selected_from"] = {"trial": best_trial, "val_auroc_mean": float(df["val_auroc_mean"].iloc[0]),
                                 "default_val_auroc_mean": float(default_row["val_auroc_mean"].iloc[0]) if len(default_row) else None}
    existing = {}
    if BEST_HYPERPARAMS_PATH.exists():
        with open(BEST_HYPERPARAMS_PATH) as f:
            existing = json.load(f)
    existing[model_name] = best_hp
    with open(BEST_HYPERPARAMS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[{model_name}] wrote best config to {BEST_HYPERPARAMS_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search hyperparameters for the autoencoders (val only)")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CLASSES), default=sorted(MODEL_CLASSES))
    parser.add_argument("--trials", type=int, default=None, help="Trials per model (default: dense 60, lstm 40, rnn 40)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel worker processes")
    parser.add_argument("--threads", type=int, default=2, help="Torch threads per worker")
    parser.add_argument("--seed", type=int, default=0, help="Search RNG seed")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(VAL_ANOMALY_DIR, "manifest.json")):
        raise SystemExit("Missing val anomaly set. Run: python generate_anomalies.py --split val")

    device_ids = sorted(_load_val_manifest().keys())
    for name in args.models:
        tune_model(name, args.trials or DEFAULT_TRIALS[name], args.workers, args.threads, args.seed, device_ids)


if __name__ == "__main__":
    main()
