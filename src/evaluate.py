"""
src/evaluate.py - Evaluation metrics for candidate classification,
observation-level Top-1 accuracy, and physical route continuity.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


def compute_candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Candidate-level classification metrics:
    Accuracy, Precision, Recall, Macro F1, and Binary Log Loss.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    # Clip probabilities to avoid log(0) numerical issues
    y_prob_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    loss = float(log_loss(y_true, y_prob_clipped))

    return {
        "candidate_accuracy": acc,
        "candidate_precision": prec,
        "candidate_recall": rec,
        "candidate_macro_f1": macro_f1,
        "candidate_binary_f1": binary_f1,
        "log_loss": loss,
    }


def compute_observation_top1(
    df: pd.DataFrame,
    prob_col: str = "prob",
    y_col: str = "y",
) -> float:
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, obs_idx), choose the candidate with highest predicted probability.
    Count as correct if the chosen candidate has y == 1 (matches the aligned ground-truth arc).
    """
    # Sort candidates by predicted probability descending within each observation
    idx_max = df.groupby(["trip_id", "obs_idx"])[prob_col].idxmax()
    top1_choices = df.loc[idx_max]
    top1_acc = float(top1_choices[y_col].mean())
    return top1_acc


def compute_route_continuity(
    df: pd.DataFrame,
    trip_arcs_dict: Dict[str, pd.DataFrame],
    prob_col: str = "prob",
) -> float:
    """
    Calculate Route Continuity across all trips in df:
    Fraction of consecutive predicted arcs (across consecutive GPS observations) that are either:
    1. Identical (vehicle remains on same segment: arc_{i} == arc_{i+1})
    2. Directly connected in the road network: arcs[arc_i].to_node == arcs[arc_{i+1}].from_node.
    """
    idx_max = df.groupby(["trip_id", "obs_idx"])[prob_col].idxmax()
    top1_df = df.loc[idx_max].sort_values(["trip_id", "obs_idx"])

    total_transitions = 0
    valid_transitions = 0

    for tid, group in top1_df.groupby("trip_id"):
        pred_arcs = group["arc_id"].to_numpy()
        if len(pred_arcs) <= 1:
            continue

        arcs_df = trip_arcs_dict[str(tid)]
        from_nodes = arcs_df["from_node"].to_numpy()
        to_nodes = arcs_df["to_node"].to_numpy()

        curr_arcs = pred_arcs[:-1]
        next_arcs = pred_arcs[1:]

        is_same = (curr_arcs == next_arcs)
        is_connected = (to_nodes[curr_arcs] == from_nodes[next_arcs])
        is_valid = is_same | is_connected

        total_transitions += len(is_valid)
        valid_transitions += int(np.sum(is_valid))

    if total_transitions == 0:
        return 0.0
    return float(valid_transitions / total_transitions) * 100.0


def compute_confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return dictionary with confusion matrix counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
