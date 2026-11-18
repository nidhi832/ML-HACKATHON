"""
PathSnap: Model Comparison Visualizations & Diagnostic Plots
Generates all publication-quality figures from real saved experiment metrics.
Zero fabricated numbers: derived strictly from results/*.csv and results/*.json.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Fix Windows matplotlib cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_cv_macro_f1(df: pd.DataFrame):
    """Plot 5-Fold GroupKFold Macro F1 with standard deviation error bars."""
    plt.figure(figsize=(10, 5.5))
    
    sorted_df = df.sort_values(by="cv_macro_f1_mean", ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sorted_df)))
    bars = plt.barh(
        sorted_df["model_name"],
        sorted_df["cv_macro_f1_mean"],
        xerr=sorted_df["cv_macro_f1_std"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    
    plt.xlabel("Validation Macro F1 Score (5-Fold GroupKFold by Trip)", fontsize=11, fontweight="bold")
    plt.ylabel("Model Family", fontsize=11, fontweight="bold")
    plt.title("PathSnap: Model Cross-Validation Macro F1 Comparison (Mean ± Std)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.xlim(0.5, 1.0)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="semibold"
        )
        
    plt.tight_layout()
    out_path = FIGURES_DIR / "cv_macro_f1_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_student_test_performance(df: pd.DataFrame):
    """Grouped bar chart for Student-Test Top-1 Observation Accuracy & Macro F1."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    sorted_df = df.sort_values(by="student_top1_acc", ascending=False)
    x = np.arange(len(sorted_df))
    width = 0.35
    
    top1_vals = sorted_df["student_top1_acc"] / 100.0 if sorted_df["student_top1_acc"].max() > 1.0 else sorted_df["student_top1_acc"]
    f1_vals = sorted_df["student_macro_f1"]
    
    rects1 = ax.bar(x - width/2, top1_vals, width, label="Top-1 Observation Accuracy", color="#2b5c8f", edgecolor="black")
    rects2 = ax.bar(x + width/2, f1_vals, width, label="Candidate Macro F1", color="#e27c38", edgecolor="black")
    
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Student-Test Performance: Top-1 Accuracy vs Candidate Macro F1", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["model_name"], rotation=15, ha="right", fontsize=10, fontweight="semibold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    out_path = FIGURES_DIR / "student_test_top1_macro_f1.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_log_loss(df: pd.DataFrame):
    """Bar chart of candidate log loss on Student-Test."""
    plt.figure(figsize=(9, 5))
    sorted_df = df.sort_values(by="log_loss", ascending=True)
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_df)))
    bars = plt.bar(
        sorted_df["model_name"],
        sorted_df["log_loss"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    
    plt.ylabel("Candidate Log Loss (Lower is Better)", fontsize=11, fontweight="bold")
    plt.title("Candidate Probability Calibration & Log Loss on Student-Test", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=20, ha="right", fontsize=10, fontweight="semibold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )
        
    plt.tight_layout()
    out_path = FIGURES_DIR / "student_test_log_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_train_val_gap(df: pd.DataFrame):
    """Diagnose over/underfitting: Train Macro F1 vs Validation Macro F1 and their Gap."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    sorted_df = df.sort_values(by="train_val_gap", ascending=False)
    x = np.arange(len(sorted_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, sorted_df["cv_train_macro_f1"], width, label="Train Macro F1", color="#3b9ab2", edgecolor="black")
    rects2 = ax.bar(x + width/2, sorted_df["cv_macro_f1_mean"], width, label="CV Val Macro F1", color="#e4b849", edgecolor="black")
    
    ax.set_ylabel("Macro F1", fontsize=11, fontweight="bold")
    ax.set_title("Overfitting Diagnosis: Train vs Validation Macro F1 and Generalization Gap", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["model_name"], rotation=15, ha="right", fontsize=10, fontweight="semibold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for idx, row in sorted_df.reset_index(drop=True).iterrows():
        gap = row["train_val_gap"]
        val_f1 = row["cv_macro_f1_mean"]
        ax.annotate(f"Gap: +{gap:.3f}",
                    xy=(idx, max(row["cv_train_macro_f1"], val_f1)),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight="bold",
                    color="#c0392b" if gap > 0.04 else "#27ae60")

    plt.tight_layout()
    out_path = FIGURES_DIR / "train_vs_val_generalization_gap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_matrices(cm_path: Path):
    """Plot 2x3 confusion matrix grid for all 6 models."""
    if not cm_path.exists():
        print(f"Confusion matrix file not found: {cm_path}")
        return
        
    with open(cm_path, "r") as f:
        cms = json.load(f)
        
    model_names = list(cms.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    
    for i, name in enumerate(model_names):
        ax = axes[i]
        cm_dict = cms[name]
        cm_data = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ], dtype=np.float64)
        row_sums = cm_data.sum(axis=1)[:, np.newaxis]
        cm_norm = np.zeros_like(cm_data)
        np.divide(cm_data, row_sums, out=cm_norm, where=row_sums != 0)
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        
        for row in range(2):
            for col in range(2):
                ax.text(
                    col + 0.5, row + 0.72,
                    f"(n={cm_data[row, col]:,})",
                    ha="center", va="center",
                    fontsize=8.5, color="gray"
                )
                
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9.5)
        ax.set_ylabel("True", fontsize=9.5)
        
    plt.suptitle("PathSnap: Normalized Confusion Matrices (Student-Test Candidates)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrices_all_models.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mlp_loss_history(loss_history_path: Path):
    """Plot epoch-wise training loss history for PyTorch MLP."""
    if not loss_history_path.exists():
        print(f"MLP loss history file not found: {loss_history_path}")
        return
        
    with open(loss_history_path, "r") as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        epochs = data.get("epochs", list(range(1, len(data.get("loss", [])) + 1)))
        train_losses = data.get("loss", [])
        val_losses = data.get("val_loss", [])
    else:
        epochs = [item["epoch"] for item in data]
        train_losses = [item["train_loss"] for item in data]
        val_losses = [item.get("val_loss", None) for item in data]
    
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker='o', linewidth=2, color="#2980b9", label="Training Loss (BCEWithLogits)")
    if any(v is not None for v in val_losses):
        valid_vals = [v for v in val_losses if v is not None]
        plt.plot(epochs[:len(valid_vals)], valid_vals, marker='s', linewidth=2, color="#e74c3c", label="Validation Loss")
        
    plt.title("PyTorch Deep MLP: Epoch-Wise Training Convergence", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", fontsize=10)
    
    final_loss = train_losses[-1]
    plt.annotate(f"Final Loss: {final_loss:.4f}",
                 xy=(epochs[-1], final_loss),
                 xytext=(-30, 20),
                 textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontweight="bold")
                 
    plt.tight_layout()
    out_path = FIGURES_DIR / "mlp_training_loss_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 80)
    print("PathSnap: Generating Figures & Diagnostic Plots")
    print("=" * 80)
    
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist yet.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded results for {len(df)} models:")
    print(df[["model_name", "cv_macro_f1_mean", "student_top1_acc", "student_macro_f1", "route_continuity_pct"]].to_string(index=False))
    
    plot_cv_macro_f1(df)
    plot_student_test_performance(df)
    plot_log_loss(df)
    plot_train_val_gap(df)
    plot_confusion_matrices(RESULTS_DIR / "confusion_matrices.json")
    plot_mlp_loss_history(RESULTS_DIR / "mlp_epoch_loss_history.json")
    
    print("\nAll figures successfully generated and saved to results/figures/.")


if __name__ == "__main__":
    main()
