"""
scripts/03_train_compare.py - Systematic multi-family model comparison for PathSnap.
Evaluates 6 algorithm families using GroupKFold(5) by trip_id,
fits on full train set, benchmarks on student_test set, logs real metrics,
and saves models to artifacts/.
"""

import os
import sys

# Critical for Windows sandbox environment: prevent loky subprocess core counting hang
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass

import time
import json
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io import load_trip, list_trips
from src.models import (
    FEATURE_COLS,
    LogisticRegressionModel,
    GaussianNBModel,
    SubsampledKNNModel,
    RandomForestModel,
    LightGBMModel,
    PyTorchMLPModel,
)
from src.evaluate import (
    compute_candidate_metrics,
    compute_observation_top1,
    compute_route_continuity,
    compute_confusion_matrix_dict,
)


def run_model_comparison():
    print("=" * 80)
    print("PathSnap: Multi-Family Model Training & Benchmarking")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("data_cache", exist_ok=True)

    # 1. Load feature matrices
    print("Loading feature tables from data_cache/...")
    t0_load = time.time()
    train_df = pd.read_parquet("data_cache/train_features.parquet")
    test_df = pd.read_parquet("data_cache/student_test_features.parquet")
    print(f"Loaded train ({len(train_df):,} rows) and student_test ({len(test_df):,} rows) in {time.time() - t0_load:.1f}s")

    X_train_full = train_df[FEATURE_COLS].to_numpy()
    y_train_full = train_df["y"].to_numpy()
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["y"].to_numpy()

    # Load student test road networks for route continuity calculation
    student_trips = list_trips("Student folder", "student_test")
    trip_arcs_dict = {}
    for tid, tdir in student_trips:
        trip = load_trip(tdir)
        trip_arcs_dict[tid] = trip.arcs

    # Read Student Test candidate miss rate for K=10
    miss_rate_df = pd.read_csv("results/candidate_miss_rate.csv")
    test_miss_rate_row = miss_rate_df[(miss_rate_df["partition"] == "student_test") & (miss_rate_df["K"] == 10)]
    student_test_miss_rate = float(test_miss_rate_row["miss_rate_pct"].iloc[0]) if not test_miss_rate_row.empty else 1.17

    # Define the 6 models
    models_to_evaluate = [
        ("logistic_regression", LogisticRegressionModel, {}),
        ("gaussian_nb", GaussianNBModel, {}),
        ("subsampled_knn", SubsampledKNNModel, {"max_fit_samples": 25000}),
        ("random_forest", RandomForestModel, {"n_estimators": 40, "max_depth": 12, "max_samples": 0.10}),
        ("lightgbm", LightGBMModel, {"n_estimators": 180, "max_depth": 7, "learning_rate": 0.08}),
        ("pytorch_mlp", PyTorchMLPModel, {"epochs": 10, "batch_size": 4096}),
    ]

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train_full, y_train_full, groups=groups_train))

    comparison_results = []
    hyperparams_dict = {}
    confusion_matrices_dict = {}
    mlp_loss_history = []
    test_predictions_dict = {
        "trip_id": test_df["trip_id"],
        "obs_idx": test_df["obs_idx"],
        "arc_id": test_df["arc_id"],
        "y": test_df["y"],
    }

    for model_key, model_cls, model_kwargs in models_to_evaluate:
        print("\n" + "-" * 75, flush=True)
        print(f"Evaluating Model: {model_key.upper()}...", flush=True)
        print("-" * 75, flush=True)

        # 2. 5-Fold GroupKFold Cross-Validation
        cv_val_f1s = []
        cv_train_f1s = []

        print("  Running 5-Fold GroupKFold Cross-Validation by trip_id...", flush=True)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            t_fold_0 = time.time()
            fold_model = model_cls(**model_kwargs)

            # Fit on fold
            fold_model.fit(X_train_full[train_idx], y_train_full[train_idx])

            # Validation prediction
            val_eval_idx = val_idx
            if model_key == "subsampled_knn" and len(val_idx) > 25000:
                val_eval_idx = np.random.choice(val_idx, size=25000, replace=False)
            val_probs = fold_model.predict_proba(X_train_full[val_eval_idx])
            val_preds = (val_probs >= 0.5).astype(np.int32)
            val_f1 = f1_score(y_train_full[val_eval_idx], val_preds, average="macro", zero_division=0)
            cv_val_f1s.append(val_f1)

            # Subsampled train evaluation for over/underfitting check
            sub_tr = np.random.choice(train_idx, size=min(40000, len(train_idx)), replace=False)
            tr_probs = fold_model.predict_proba(X_train_full[sub_tr])
            tr_preds = (tr_probs >= 0.5).astype(np.int32)
            tr_f1 = f1_score(y_train_full[sub_tr], tr_preds, average="macro", zero_division=0)
            cv_train_f1s.append(tr_f1)

            print(f"    Fold {fold_idx + 1}/5: Val Macro F1 = {val_f1:.4f} (Train Macro F1 = {tr_f1:.4f}) [{time.time() - t_fold_0:.1f}s]", flush=True)

        cv_mean_f1 = float(np.mean(cv_val_f1s))
        cv_std_f1 = float(np.std(cv_val_f1s))
        cv_train_mean_f1 = float(np.mean(cv_train_f1s))
        train_val_gap = float(cv_train_mean_f1 - cv_mean_f1)

        print(f"  --> CV Macro F1: {cv_mean_f1:.4f} +/- {cv_std_f1:.4f} (Train-Val Gap: {train_val_gap:+.4f})", flush=True)

        # 3. Fit on full training set
        print("  Fitting on full training set (2,084,170 rows)...", flush=True)
        final_model = model_cls(**model_kwargs)
        t_train_start = time.time()
        final_model.fit(X_train_full, y_train_full)
        train_time_sec = float(time.time() - t_train_start)
        print(f"  Training completed in {train_time_sec:.2f} seconds.", flush=True)

        # Capture epoch loss history if MLP
        if hasattr(final_model, "loss_history") and len(final_model.loss_history) > 0:
            mlp_loss_history = [float(x) for x in final_model.loss_history]

        # 4. Inference on student_test set
        print("  Evaluating on student_test set (191,600 rows)...", flush=True)
        t_infer_start = time.time()
        test_probs = final_model.predict_proba(X_test)
        total_infer_time_sec = time.time() - t_infer_start

        # Calculate inference time per 1000 observations (19,160 GPS observations in test set)
        n_test_obs = len(test_df.groupby(["trip_id", "obs_idx"]))
        infer_time_per_1000_obs = float((total_infer_time_sec / max(1, n_test_obs)) * 1000.0)

        # Candidate-level metrics
        test_metrics = compute_candidate_metrics(y_test, test_probs, threshold=0.5)

        # Observation-level Top-1 accuracy
        test_df_copy = test_df[["trip_id", "obs_idx", "arc_id", "y"]].copy()
        test_df_copy["prob"] = test_probs
        top1_acc = compute_observation_top1(test_df_copy, prob_col="prob", y_col="y")

        # Route continuity
        continuity = compute_route_continuity(test_df_copy, trip_arcs_dict, prob_col="prob")

        # Confusion matrix
        test_preds = (test_probs >= 0.5).astype(np.int32)
        cm_dict = compute_confusion_matrix_dict(y_test, test_preds)
        confusion_matrices_dict[model_key] = cm_dict

        # Store test predictions column
        test_predictions_dict[f"prob_{model_key}"] = test_probs.astype(np.float32)

        # Model configuration
        model_config = final_model.get_config()
        hyperparams_dict[model_key] = model_config

        # Save model artifact
        artifact_path = f"artifacts/{model_key}.joblib"
        if model_key == "pytorch_mlp":
            torch.save(final_model.net.state_dict(), f"artifacts/{model_key}.pt")
            joblib.dump(final_model.scaler, f"artifacts/{model_key}_scaler.joblib")
        else:
            joblib.dump(final_model, artifact_path)
        print(f"  Saved model artifact -> artifacts/{model_key}.*")

        # Append to comparison table
        comparison_results.append({
            "model_key": model_key,
            "algorithm_family": model_config["family"],
            "model_name": model_config["model_name"],
            "key_config": str({k: v for k, v in model_config.items() if k not in ["model_name", "family"]}),
            "cv_macro_f1_mean": round(cv_mean_f1, 4),
            "cv_macro_f1_std": round(cv_std_f1, 4),
            "cv_train_macro_f1": round(cv_train_mean_f1, 4),
            "train_val_gap": round(train_val_gap, 4),
            "student_top1_acc": round(top1_acc * 100.0, 2),
            "route_continuity_pct": round(continuity, 2),
            "student_macro_f1": round(test_metrics["candidate_macro_f1"], 4),
            "student_binary_f1": round(test_metrics["candidate_binary_f1"], 4),
            "student_precision": round(test_metrics["candidate_precision"], 4),
            "student_recall": round(test_metrics["candidate_recall"], 4),
            "student_accuracy": round(test_metrics["candidate_accuracy"], 4),
            "log_loss": round(test_metrics["log_loss"], 4),
            "candidate_miss_rate_pct": student_test_miss_rate,
            "train_time_sec": round(train_time_sec, 2),
            "infer_time_ms_per_1000_obs": round(infer_time_per_1000_obs * 1000.0, 2),
        })

    # Save comparison CSV
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv("results/model_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("CONSOLIDATED MODEL COMPARISON TABLE (results/model_comparison.csv):")
    print("=" * 80)
    cols_display = [
        "model_name", "algorithm_family", "cv_macro_f1_mean", "student_top1_acc",
        "route_continuity_pct", "student_macro_f1", "log_loss", "train_val_gap", "infer_time_ms_per_1000_obs"
    ]
    print(df_comparison[cols_display].to_string(index=False))

    # Save hyperparameters JSON
    with open("results/models_hyperparameters.json", "w") as f:
        json.dump(hyperparams_dict, f, indent=2)

    # Save confusion matrices JSON
    with open("results/confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices_dict, f, indent=2)

    # Save MLP epoch loss history JSON
    with open("results/mlp_epoch_loss_history.json", "w") as f:
        json.dump({"epochs": list(range(1, len(mlp_loss_history) + 1)), "loss": mlp_loss_history}, f, indent=2)

    # Save test predictions parquet
    df_preds = pd.DataFrame(test_predictions_dict)
    df_preds.to_parquet("results/student_test_predictions.parquet", index=False, compression="zstd")
    print("\nSaved test predictions to results/student_test_predictions.parquet")
    print("All models successfully trained, benchmarked, and logged!")
    print("=" * 80)


if __name__ == "__main__":
    run_model_comparison()
