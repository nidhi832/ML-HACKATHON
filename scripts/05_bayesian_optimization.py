"""
scripts/05_bayesian_optimization.py - Bayesian Hyperparameter Optimisation for Champion Model.
Implements Bayesian search using Optuna (TPESampler) on 5-Fold GroupKFold by trip_id.
Saves trial history to results/bayesian_optimization_trials.csv and best parameters.
"""

import os
import sys
# Set matplotlib cache directory to temp before importing pyplot
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import FEATURE_COLS

DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Suppress Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(n_trials: int = 15):
    print("=" * 80)
    print("PathSnap: Bayesian Hyperparameter Optimisation (LightGBM TPESampler)")
    print("=" * 80)

    train_path = DATA_CACHE_DIR / "train_features.parquet"
    test_path = DATA_CACHE_DIR / "student_test_features.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Feature parquets missing. Run 02_build_dataset.py first.")

    print("Loading feature datasets...", flush=True)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Subsample trips for efficient Bayesian search while preserving group integrity
    unique_trips = np.unique(train_df["trip_id"])
    np.random.seed(42)
    sample_trips = np.random.choice(unique_trips, size=min(20, len(unique_trips)), replace=False)
    search_df = train_df[train_df["trip_id"].isin(sample_trips)].copy()

    feature_cols = FEATURE_COLS
    X = search_df[feature_cols].values
    y = search_df["y"].values.astype(int)
    groups = search_df["trip_id"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["y"].values.astype(int)

    # Use 3-fold group CV during Bayesian search for speed while preserving group isolation
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(X, y, groups=groups))

    print(f"Search Subset: {len(search_df):,} candidates across {len(sample_trips)} trips", flush=True)
    print(f"Features: {len(feature_cols)} dimensions | Optimisation Objective: GroupKFold CV Macro F1", flush=True)
    print("-" * 80, flush=True)

    trials_data = []

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 2,
            "n_estimators": 40,
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        }

        fold_f1s = []
        t0 = time.time()
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            fold_f1s.append(f1_score(y_val, y_pred, average="macro"))

        mean_f1 = float(np.mean(fold_f1s))
        duration = round(time.time() - t0, 2)

        trial_record = {
            "trial_number": trial.number,
            "cv_macro_f1": round(mean_f1, 4),
            "duration_s": duration,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in params.items() if k not in ["objective", "metric", "boosting_type", "random_state", "verbosity", "n_jobs", "n_estimators"]}
        }
        trials_data.append(trial_record)
        print(f"Trial {trial.number:2d}/{n_trials} | CV Macro F1: {mean_f1:.4f} | leaves: {params['num_leaves']} | depth: {params['max_depth']} | lr: {params['learning_rate']:.4f} | ({duration}s)", flush=True)
        return mean_f1

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print("-" * 80)
    print(f"Optimisation Complete! Best Trial #{best_trial.number}: CV Macro F1 = {best_trial.value:.4f}")
    print("Optimal Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  • {k}: {v}")

    # Save trials CSV
    df_trials = pd.DataFrame(trials_data)
    trials_csv_path = RESULTS_DIR / "bayesian_optimization_trials.csv"
    df_trials.to_csv(trials_csv_path, index=False)
    print(f"Saved: {trials_csv_path}")

    # Save best parameters JSON
    best_json_path = RESULTS_DIR / "bayesian_best_params.json"
    with open(best_json_path, "w") as f:
        json.dump({
            "best_trial_number": best_trial.number,
            "best_cv_macro_f1": round(best_trial.value, 4),
            "n_trials": n_trials,
            "search_space": {
                "num_leaves": [31, 127],
                "max_depth": [5, 12],
                "learning_rate": [0.03, 0.15],
                "subsample": [0.6, 0.95],
                "colsample_bytree": [0.6, 0.95],
                "min_child_samples": [20, 100]
            },
            "best_params": best_trial.params
        }, f, indent=2)
    print(f"Saved: {best_json_path}")

    # Generate Optimization Convergence Plot
    plt.figure(figsize=(9, 5))
    trial_nums = df_trials["trial_number"].values
    f1_scores = df_trials["cv_macro_f1"].values
    running_max = np.maximum.accumulate(f1_scores)

    plt.scatter(trial_nums, f1_scores, color="#3498db", label="Trial CV Macro F1", alpha=0.75, s=60)
    plt.plot(trial_nums, running_max, color="#e74c3c", linestyle="--", linewidth=2.2, label="Best Cumulative F1")
    plt.title("Bayesian Hyperparameter Optimisation: Convergence History (LightGBM)", fontsize=13, fontweight="bold")
    plt.xlabel("Trial Number", fontsize=11)
    plt.ylabel("CV Macro F1-Score", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_path = FIGURES_DIR / "bayesian_optimization_history.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved convergence figure: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=10)
