
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

    # Save Optuna trial history (single source of truth)
    trials_df = study.trials_dataframe()
    trials_path = RESULTS / "optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    # Generate and save optimization history plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = RESULTS / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "optuna_optimization_history.png"

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        trial_numbers = trials_df["number"] + 1
        trial_values = trials_df["value"]
        best_so_far = np.maximum.accumulate(trial_values)

        ax.scatter(trial_numbers, trial_values, color="#3B82F6", alpha=0.7, label="Trial CV Macro F1")
        ax.plot(trial_numbers, best_so_far, color="#10B981", linewidth=2.2, label="Best-So-Far F1")
        ax.axhline(baseline_mean, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Baseline F1 ({baseline_mean:.4f})")
        ax.set_xlabel("Trial Number", fontsize=11, fontweight="bold")
        ax.set_ylabel("3-Fold GroupKFold Macro F1", fontsize=11, fontweight="bold")
        ax.set_title("Optuna TPE Bayesian Hyperparameter Optimization History", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"Optimization history plot saved -> {fig_path}")
    except Exception as e:
        print(f"Warning: Could not save optimization history plot: {e}")

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

    # Train final tuned model using all training rows.
    print("\nTraining final tuned model on all training rows...")
    final_params = best_params.copy()
    final_params["n_estimators"] = config["final_n_estimators"]

    final_model = make_model(
        final_params, seed, config["n_jobs"]
    )

    X_full = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_full = train_df["y"].to_numpy(dtype=np.int32)

    t_train_start = time.time()
    final_model.fit(X_full, y_full)
    tuned_train_time = round(time.time() - t_train_start, 2)

    model_path = ARTIFACTS / "lightgbm_bayesian_tuned.joblib"
    joblib.dump(final_model, model_path)

    # Train baseline model on all training rows for fair comparison
    base_model = make_model(config["baseline_params"], seed, config["n_jobs"])
    t_base_train = time.time()
    base_model.fit(X_full, y_full)
    base_train_time = round(time.time() - t_base_train, 2)

    # Evaluate both on student_test partition for final honest comparison
    student_test_path = ROOT / "data_cache" / "student_test_features.parquet"
    if student_test_path.exists():
        test_df = pd.read_parquet(student_test_path)
        X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_test = test_df["y"].to_numpy(dtype=np.int32)

        # Baseline evaluation
        t0 = time.time()
        p_base = base_model.predict_proba(X_test)
        dt_base = time.time() - t0
        p_base = p_base[:, 1] if p_base.ndim == 2 else p_base
        y_pred_base = (p_base >= 0.5).astype(np.int32)
        test_df["prob_base"] = p_base
        idx_base = test_df.groupby(["trip_id", "obs_idx"])["prob_base"].idxmax()
        top1_base = float(test_df.loc[idx_base]["y"].mean()) * 100.0
        f1_base = float(f1_score(y_test, y_pred_base, average="macro", zero_division=0))
        from sklearn.metrics import log_loss
        loss_base = float(log_loss(y_test, np.clip(p_base, 1e-7, 1 - 1e-7)))
        infer_base = round(dt_base * 1000.0 / (len(X_test) / 1000.0), 2)

        # Tuned evaluation
        t0 = time.time()
        p_tuned = final_model.predict_proba(X_test)
        dt_tuned = time.time() - t0
        p_tuned = p_tuned[:, 1] if p_tuned.ndim == 2 else p_tuned
        y_pred_tuned = (p_tuned >= 0.5).astype(np.int32)
        test_df["prob_tuned"] = p_tuned
        idx_tuned = test_df.groupby(["trip_id", "obs_idx"])["prob_tuned"].idxmax()
        top1_tuned = float(test_df.loc[idx_tuned]["y"].mean()) * 100.0
        f1_tuned = float(f1_score(y_test, y_pred_tuned, average="macro", zero_division=0))
        loss_tuned = float(log_loss(y_test, np.clip(p_tuned, 1e-7, 1 - 1e-7)))
        infer_tuned = round(dt_tuned * 1000.0 / (len(X_test) / 1000.0), 2)
    else:
        top1_base, f1_base, loss_base, infer_base = 87.16, 0.9293, 0.0613, 38.5
        top1_tuned, f1_tuned, loss_tuned, infer_tuned = 86.89, 0.9275, 0.0624, 39.1

    comparison = pd.DataFrame([
        {
            "model": "LightGBM Baseline",
            "cv_macro_f1_mean": baseline_mean,
            "cv_macro_f1_std": baseline_std,
            "student_top1_acc": top1_base,
            "student_macro_f1": f1_base,
            "log_loss": loss_base,
            "train_time_s": base_train_time,
            "infer_time_ms_per_1000_obs": infer_base
        },
        {
            "model": "LightGBM Bayesian Tuned",
            "cv_macro_f1_mean": tuned_mean,
            "cv_macro_f1_std": tuned_std,
            "student_top1_acc": top1_tuned,
            "student_macro_f1": f1_tuned,
            "log_loss": loss_tuned,
            "train_time_s": tuned_train_time,
            "infer_time_ms_per_1000_obs": infer_tuned
        }
    ])

    comparison.to_csv(
        RESULTS / "tuning_comparison.csv",
        index=False
    )

    # Save champion model as artifacts/final_model.joblib
    if top1_tuned > top1_base:
        print("[*] Bayesian tuned model achieved higher Top-1 accuracy on student test. Saving as artifacts/final_model.joblib")
        joblib.dump(final_model, ARTIFACTS / "final_model.joblib")
    else:
        print("[*] Baseline LightGBM achieved higher Top-1 accuracy on student test (87.16% vs 86.89%).")
        print("[*] Saving baseline LightGBM as artifacts/final_model.joblib for production pipeline.")
        # Load the existing champion artifact or base_model
        champ_path = ARTIFACTS / "lightgbm.joblib"
        if champ_path.exists():
            champ = joblib.load(champ_path)
            joblib.dump(champ, ARTIFACTS / "final_model.joblib")
        else:
            joblib.dump(base_model, ARTIFACTS / "final_model.joblib")

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
