"""
src/scoring.py - Route and Trajectory Evaluation for PathSnap.

Evaluates predicted routes against reference routes strictly where evaluation
is authorized (e.g. training and student-testing partitions).

CRITICAL DATA SEPARATION:
The prediction pipeline (src/pipeline.py) and API endpoints (/predict) NEVER
access reference routes (.route) during prediction.
Scoring logic is strictly isolated in this module and only called AFTER prediction
when ground truth is available.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_io import load_trip, TripData
from src.labels import align_gps_to_route
from src.evaluate import (
    compute_candidate_metrics,
    compute_confusion_matrix_dict,
)
from src.hmm import compute_sequence_metrics


def evaluate_trip_prediction(
    trip_id: Optional[str] = None,
    predicted_arcs: Optional[List[int]] = None,
    trip_dir: Optional[str] = None,
    data_root: str = "Student folder",
    partition: str = "student_test",
    candidates_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Evaluate predicted arc sequence against the reference route for a given trip.
    
    Args:
        trip_id: Trip identifier (e.g. '00000052')
        predicted_arcs: List of predicted arc IDs, one per GPS observation.
        trip_dir: Direct path to trip directory (optional)
        data_root: Root dataset folder.
        partition: 'student_test' or 'train'.
        candidates_df: Optional DataFrame with candidate features and 'prob' column.
        
    Returns:
        Structured evaluation metrics dictionary including:
        - top1_accuracy_pct
        - route_continuity_pct
        - route_lcs_f1, route_set_jaccard
        - observation counts
        - candidate classification metrics if candidates_df provided
    """
    if trip_dir is not None:
        resolved_dir = str(trip_dir)
        if trip_id is None:
            trip_id = os.path.basename(resolved_dir)
    elif trip_id is not None:
        resolved_dir = os.path.join(PROJECT_ROOT, data_root, partition, trip_id)
        if not os.path.isdir(resolved_dir):
            alt_dir = os.path.join(data_root, partition, trip_id)
            if os.path.isdir(alt_dir):
                resolved_dir = alt_dir
            else:
                raise FileNotFoundError(f"Trip directory not found: {resolved_dir}")
    else:
        raise ValueError("Either trip_id or trip_dir must be provided.")

    trip: TripData = load_trip(resolved_dir)
    num_obs = len(trip.track)
    arcs_array = trip.arcs.to_numpy()

    if predicted_arcs is None:
        predicted_arcs = []

    # Calculate predicted continuity from network arcs
    pred_continuity_pct = _calculate_continuity(np.asarray(predicted_arcs, dtype=np.int32), arcs_array) if len(predicted_arcs) > 0 else 0.0

    if trip.route is None or len(trip.route) == 0:
        return {
            "trip_id": str(trip_id),
            "has_reference_route": False,
            "message": "Reference route (.route) is not available for this trip partition.",
            "num_observations": num_obs,
            "top1_accuracy_pct": None,
            "route_continuity_pct": round(pred_continuity_pct, 2),
            "route_lcs_f1": None,
            "num_reference_arcs": 0,
            "num_unique_predicted_arcs": len(set(predicted_arcs)),
        }

    if len(predicted_arcs) != num_obs:
        raise ValueError(
            f"Length mismatch: predicted_arcs has {len(predicted_arcs)} elements, "
            f"but trip has {num_obs} GPS observations."
        )

    # 1. Align GPS points to reference route via monotonic dynamic programming
    aligned_df = align_gps_to_route(trip)
    ground_truth_arcs = aligned_df["aligned_arc_id"].to_numpy(dtype=np.int32)
    preds = np.asarray(predicted_arcs, dtype=np.int32)

    # 2. Observation-level Top-1 Accuracy
    is_correct = (preds == ground_truth_arcs)
    correct_count = int(np.sum(is_correct))
    top1_accuracy = float(np.mean(is_correct)) * 100.0 if num_obs > 0 else 0.0

    # 3. Reference Route Continuity
    ref_continuity_pct = _calculate_continuity(ground_truth_arcs, arcs_array)

    # 4. Sequence-level metrics (LCS Route F1 and Route Jaccard)
    seq_metrics = compute_sequence_metrics(predicted_arcs, trip.route)

    results: Dict[str, Any] = {
        "trip_id": str(trip_id),
        "has_reference_route": True,
        "num_observations": num_obs,
        "correct_observations": correct_count,
        "top1_accuracy_pct": round(top1_accuracy, 2),
        "route_continuity_pct": round(pred_continuity_pct, 2),
        "reference_continuity_pct": round(ref_continuity_pct, 2),
        "route_lcs_f1": round(seq_metrics.get("route_lcs_f1", 0.0), 4),
        "route_lcs_precision": round(seq_metrics.get("route_lcs_precision", 0.0), 4),
        "route_lcs_recall": round(seq_metrics.get("route_lcs_recall", 0.0), 4),
        "route_set_jaccard": round(seq_metrics.get("route_set_jaccard", 0.0), 4),
        "num_reference_arcs": len(trip.route),
        "num_unique_predicted_arcs": len(set(predicted_arcs)),
    }

    # 5. Candidate-level classification metrics if candidate features and probabilities are available
    if candidates_df is not None and "prob" in candidates_df.columns:
        gt_map = dict(zip(aligned_df["obs_idx"], ground_truth_arcs))
        cand_obs = candidates_df["obs_idx"].to_numpy()
        cand_arcs = candidates_df["arc_id"].to_numpy()
        cand_probs = candidates_df["prob"].to_numpy()

        y_true = np.array([1 if cand_arcs[i] == gt_map.get(cand_obs[i], -1) else 0 for i in range(len(cand_obs))], dtype=np.int32)
        cand_metrics = compute_candidate_metrics(y_true, cand_probs, threshold=0.5)
        cm_dict = compute_confusion_matrix_dict(y_true, (cand_probs >= 0.5).astype(np.int32))

        # Check candidate recall (is ground-truth arc in candidates for each obs?)
        obs_with_gt_in_cand = 0
        cand_grouped = candidates_df.groupby("obs_idx")["arc_id"].apply(set)
        for o_idx, gt_arc in gt_map.items():
            if gt_arc in cand_grouped.get(o_idx, set()):
                obs_with_gt_in_cand += 1
        cand_recall_pct = (obs_with_gt_in_cand / max(1, num_obs)) * 100.0

        results.update({
            "candidate_recall_pct": round(cand_recall_pct, 2),
            "candidate_miss_rate_pct": round(100.0 - cand_recall_pct, 2),
            "candidate_precision": round(cand_metrics["candidate_precision"], 4),
            "candidate_recall": round(cand_metrics["candidate_recall"], 4),
            "candidate_macro_f1": round(cand_metrics["candidate_macro_f1"], 4),
            "candidate_binary_f1": round(cand_metrics["candidate_binary_f1"], 4),
            "log_loss": round(cand_metrics["log_loss"], 4),
            "confusion_matrix": cm_dict,
        })

    return results


def _calculate_continuity(arcs_seq: np.ndarray, arcs_array: np.ndarray) -> float:
    """Calculate percentage of valid consecutive road transitions."""
    if len(arcs_seq) <= 1:
        return 100.0

    curr = arcs_seq[:-1]
    nxt = arcs_seq[1:]
    num_arcs = len(arcs_array)

    valid_mask = (curr < num_arcs) & (nxt < num_arcs)
    if not np.all(valid_mask):
        return 0.0

    from_nodes = arcs_array[:, 0]
    to_nodes = arcs_array[:, 1]

    is_same = (curr == nxt)
    is_connected = (to_nodes[curr] == from_nodes[nxt])
    is_valid = is_same | is_connected

    return float(np.mean(is_valid) * 100.0)
