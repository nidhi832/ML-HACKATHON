
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURE_COLS = [
    "dist_m", "proj_frac", "raw_proj_frac", "arc_length_m", "candidate_rank",
    "dist_gap_to_best_m", "n_cand_within_10m", "n_cand_within_25m",
    "n_cand_within_50m", "arc_bearing", "gps_heading", "heading_abs_diff",
    "heading_cos_diff", "speed_mps", "is_stationary", "turn_angle_deg",
    "acceleration_mps2", "dt_prev", "dt_next", "prev_point_dist_to_arc",
    "next_point_dist_to_arc", "rolling_dist_mean_3", "dist_change_prev",
    "arc_out_degree", "arc_in_degree", "has_reverse_arc"
]
CONFIG_PATH = ROOT / "configs" / "bayes_opt_config.json"
TRAIN_PATH = ROOT / "data_cache" / "train_features.parquet"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def make_model(params, random_state, n_jobs):
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        **params
    )


def evaluate_cv(params, X, y, groups, splits, random_state, n_jobs):
    scores = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = make_model(params, random_state + fold, n_jobs)
        model.fit(X[train_idx], y[train_idx])

        predictions = model.predict(X[valid_idx])
        score = f1_score(
            y[valid_idx], predictions,
            average="macro", zero_division=0
        )
        scores.append(float(score))

    return float(np.mean(scores)), float(np.std(scores))


def main():
    config = load_config()
    seed = config["random_state"]

    print("=" * 70)
    print("PART B: BAYESIAN HYPERPARAMETER OPTIMISATION")
    print("=" * 70)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Training features not found. Run scripts/02_build_dataset.py first."
        )

    print("Loading training features...")
    train_df = pd.read_parquet(TRAIN_PATH)

    required = FEATURE_COLS + ["trip_id", "y"]
    missing = [column for column in required if column not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train_df = train_df[required].dropna().copy()
    unique_trips = np.sort(train_df["trip_id"].unique())

    if len(unique_trips) < config["cv_splits"]:
        raise ValueError("Not enough distinct trips for GroupKFold.")

    rng = np.random.default_rng(seed)
    selected_trips = rng.choice(
        unique_trips,
        size=min(config["search_trip_count"], len(unique_trips)),
        replace=False
    )

    search_df = train_df[train_df["trip_id"].isin(selected_trips)].copy()

    X = search_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = search_df["y"].to_numpy(dtype=np.int32)
    groups = search_df["trip_id"].to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError("Search data must contain both label classes.")

    splits = list(
        GroupKFold(n_splits=config["cv_splits"]).split(X, y, groups)
    )

    print(f"Training rows: {len(train_df):,}")
    print(f"Search rows: {len(search_df):,}")
    print(f"Search trips: {len(selected_trips)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Cross-validation folds: {len(splits)}")
    print(f"Optuna trials: {config['n_trials']}")

    # Baseline evaluation on exactly the same search data and folds.
    print("\nEvaluating baseline LightGBM...")
    baseline_params = config["baseline_params"]
    baseline_mean, baseline_std = evaluate_cv(
        baseline_params, X, y, groups, splits, seed, config["n_jobs"]
    )

    print(
        f"Baseline CV Macro F1: {baseline_mean:.5f} "
        f"+/- {baseline_std:.5f}"
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                config["search_n_estimators_min"],
                config["search_n_estimators_max"]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.15, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 0.95
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 100
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            )
        }

        started = time.time()
        mean_score, std_score = evaluate_cv(
            params, X, y, groups, splits, seed, config["n_jobs"]
        )
        duration = time.time() - started

        trial.set_user_attr("cv_std", std_score)
        trial.set_user_attr("duration_seconds", round(duration, 2))

        print(
            f"Trial {trial.number + 1}/{config['n_trials']} | "
            f"Macro F1: {mean_score:.5f} | "
            f"Time: {duration:.1f}s",
            flush=True
        )
        return mean_score

    print("\nStarting Optuna TPE search...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=config["n_trials"])

    best_params = dict(study.best_params)
    best_params["subsample_freq"] = 1

    print("\nBest trial:", study.best_trial.number)
    print("Best CV Macro F1:", round(study.best_value, 6))
    print("Best parameters:", best_params)

    # Save Optuna trial history.
    trials_df = study.trials_dataframe()
    trials_path = RESULTS / "optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    # Preserve the existing project naming convention too.
    trials_df.to_csv(
        RESULTS / "bayesian_optimization_trials.csv",
        index=False
    )

    best_info = {
        "best_trial_number": study.best_trial.number,
        "best_cv_macro_f1": float(study.best_value),
        "n_trials": config["n_trials"],
        "search_trips": len(selected_trips),
        "cv_folds": config["cv_splits"],
        "feature_count": len(FEATURE_COLS),
        "best_params": best_params
    }

    with open(
        RESULTS / "bayesian_best_params.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(best_info, file, indent=2)

    # Compare baseline and tuned model on the same CV folds.
    tuned_mean, tuned_std = evaluate_cv(
        best_params, X, y, groups, splits, seed, config["n_jobs"]
    )

    comparison = pd.DataFrame([
        {
            "model": "LightGBM Baseline",
            "cv_macro_f1_mean": baseline_mean,
            "cv_macro_f1_std": baseline_std
        },
        {
            "model": "LightGBM Bayesian Tuned",
            "cv_macro_f1_mean": tuned_mean,
            "cv_macro_f1_std": tuned_std
        }
    ])

    comparison.to_csv(
        RESULTS / "tuning_comparison.csv",
        index=False
    )

    # Train final tuned model using all training rows.
    print("\nTraining final tuned model on all training rows...")
    final_params = best_params.copy()
    final_params["n_estimators"] = config["final_n_estimators"]

    final_model = make_model(
        final_params, seed, config["n_jobs"]
    )

    X_full = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_full = train_df["y"].to_numpy(dtype=np.int32)

    final_model.fit(X_full, y_full)

    model_path = ARTIFACTS / "lightgbm_bayesian_tuned.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "model": "LightGBM",
        "training_rows": len(train_df),
        "features": FEATURE_COLS,
        "final_params": final_params,
        "selection_metric": "GroupKFold CV Macro F1",
        "best_search_cv_macro_f1": float(study.best_value),
        "student_test_used_for_tuning": False
    }

    with open(
        ARTIFACTS / "lightgbm_bayesian_tuned_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMISATION COMPLETE")
    print("=" * 70)
    print(f"Baseline CV Macro F1: {baseline_mean:.5f}")
    print(f"Tuned CV Macro F1:    {tuned_mean:.5f}")
    print(f"Difference:           {tuned_mean - baseline_mean:+.5f}")
    print(f"Trial log:            {trials_path}")
    print(f"Comparison:           {RESULTS / 'tuning_comparison.csv'}")
    print(f"Final model:          {model_path}")
    print(f"Metadata:             {ARTIFACTS / 'lightgbm_bayesian_tuned_metadata.json'}")
    print("Student test data was not used for tuning or model selection.")


if __name__ == "__main__":
    main()
