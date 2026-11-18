"""
scripts/test_all_trips.py - End-to-End Validation Across All 10 Student-Test Trips.

Responsibilities:
1. Iterates over all 10 trips in Student folder/student_test.
2. Sends each trip through the identical prediction pipeline used by the FastAPI API.
3. Evaluates predicted trajectories against ground-truth routes using src/scoring.py.
4. Measures and records latency, Top-1 accuracy, candidate recall, route continuity, and F1.
5. Generates a consolidated results file in results/student_test_all_trips.json.
6. Prints a clean terminal summary table.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Prevent joblib loky from calling external subprocess to count CPUs in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import list_trips
from src.pipeline import get_pipeline, MapMatchingPipeline, PipelineResult
from src.scoring import evaluate_trip_prediction


def run_all_trips_test(
    data_root: str = "Student folder",
    partition: str = "student_test",
    output_json: str = "results/student_test_all_trips.json",
) -> bool:
    print("=" * 105)
    print("   PathSnap: End-to-End Pipeline Evaluation on Student-Test Dataset (Person C)")
    print("=" * 105)

    data_dir = PROJECT_ROOT / data_root
    trips_list = list_trips(str(data_dir), partition)
    total_trips = len(trips_list)
    print(f"[*] Found {total_trips} trips in partition '{partition}'.")

    # Initialize prediction pipeline
    t_init_start = time.perf_counter()
    pipeline: MapMatchingPipeline = get_pipeline()
    init_time_ms = (time.perf_counter() - t_init_start) * 1000.0
    print(f"[*] Pipeline initialized (Model: {pipeline.model_name}, Decoder: HMM Viterbi) in {init_time_ms:.1f}ms\n")

    results_list: List[Dict[str, Any]] = []
    successful_trips = 0
    failed_trips = 0

    t_total_start = time.perf_counter()

    for idx, (trip_id, trip_dir) in enumerate(trips_list, start=1):
        print(f"[{idx:02d}/{total_trips:02d}] Processing trip '{trip_id}'...", end=" ", flush=True)

        try:
            # 1. Run inference through prediction pipeline (NEVER touches .route)
            pipe_result: PipelineResult = pipeline.predict_trip(
                trip_id=trip_id,
                data_root=data_root,
                partition=partition,
                return_candidates_df=True,
            )

            # 2. Evaluate against reference route using scoring module
            eval_metrics = evaluate_trip_prediction(
                trip_id=trip_id,
                predicted_arcs=pipe_result.predicted_arcs,
                data_root=data_root,
                partition=partition,
                candidates_df=pipe_result.candidates_df,
            )

            trip_record: Dict[str, Any] = {
                "trip_id": trip_id,
                "status": "SUCCESS",
                "num_observations": pipe_result.num_observations,
                "num_arcs_traversed": pipe_result.num_arcs_traversed,
                "top1_accuracy_pct": eval_metrics.get("top1_accuracy_pct", 0.0),
                "route_continuity_pct": pipe_result.continuity_pct,
                "reference_continuity_pct": eval_metrics.get("reference_continuity_pct", 0.0),
                "candidate_recall_pct": eval_metrics.get("candidate_recall_pct", 0.0),
                "candidate_macro_f1": eval_metrics.get("candidate_macro_f1", 0.0),
                "log_loss": eval_metrics.get("log_loss", 0.0),
                "processing_time_ms": round(pipe_result.processing_time_ms, 2),
            }

            successful_trips += 1
            print(
                f"OK -> Top-1: {trip_record['top1_accuracy_pct']:>6.2f}% | "
                f"Continuity: {trip_record['route_continuity_pct']:>6.2f}% | "
                f"Recall: {trip_record['candidate_recall_pct']:>6.2f}% | "
                f"Time: {trip_record['processing_time_ms']:>6.1f}ms"
            )

        except Exception as exc:
            failed_trips += 1
            trip_record = {
                "trip_id": trip_id,
                "status": "FAILED",
                "error": str(exc),
            }
            print(f"FAILED: {exc}")

        results_list.append(trip_record)

    total_wall_time_s = time.perf_counter() - t_total_start

    # Print summary table
    print("\n" + "=" * 105)
    print("CONSOLIDATED STUDENT-TEST TRIPS RESULTS SUMMARY")
    print("=" * 105)
    header = (
        f"{'Trip ID':<10} | {'Status':<7} | {'Points':<6} | {'Traversed':<9} | "
        f"{'Top-1 Acc':<9} | {'Continuity':<10} | {'Recall':<8} | {'Macro F1':<8} | {'Latency':<9}"
    )
    print(header)
    print("-" * 105)

    valid_top1 = []
    valid_continuity = []
    valid_recall = []
    valid_f1 = []
    valid_times = []

    for r in results_list:
        if r["status"] == "SUCCESS":
            valid_top1.append(r["top1_accuracy_pct"])
            valid_continuity.append(r["route_continuity_pct"])
            valid_recall.append(r["candidate_recall_pct"])
            valid_f1.append(r["candidate_macro_f1"])
            valid_times.append(r["processing_time_ms"])

            print(
                f"{r['trip_id']:<10} | {r['status']:<7} | {r['num_observations']:<6} | "
                f"{r['num_arcs_traversed']:<9} | {r['top1_accuracy_pct']:>8.2f}% | "
                f"{r['route_continuity_pct']:>9.2f}% | {r['candidate_recall_pct']:>7.2f}% | "
                f"{r['candidate_macro_f1']:>8.4f} | {r['processing_time_ms']:>7.1f}ms"
            )
        else:
            print(f"{r['trip_id']:<10} | {r['status']:<7} | ERROR: {r.get('error', 'Unknown')}")

    print("-" * 105)
    if valid_top1:
        mean_top1 = np.mean(valid_top1)
        mean_cont = np.mean(valid_continuity)
        mean_rec = np.mean(valid_recall)
        mean_f1 = np.mean(valid_f1)
        mean_time = np.mean(valid_times)
        total_proc_time_s = sum(valid_times) / 1000.0

        print(
            f"{'AVERAGE':<10} | {'ALL':<7} | {'-':<6} | {'-':<9} | "
            f"{mean_top1:>8.2f}% | {mean_cont:>9.2f}% | {mean_rec:>7.2f}% | "
            f"{mean_f1:>8.4f} | {mean_time:>7.1f}ms"
        )
    print("=" * 105)

    print(f"\nExecution Summary:")
    print(f"- Total Trips Tested : {total_trips}")
    print(f"- Successful Trips   : {successful_trips}")
    print(f"- Failed Trips       : {failed_trips}")
    print(f"- Total Wall Time    : {total_wall_time_s:.2f} seconds")
    if valid_top1:
        print(f"- Mean Top-1 Accuracy: {mean_top1:.2f}%")
        print(f"- Mean Continuity    : {mean_cont:.2f}%")
        print(f"- Mean Cand. Recall  : {mean_rec:.2f}%")
        print(f"- Mean Latency / Trip: {mean_time:.1f} ms")

    # Save output to JSON
    out_file = PROJECT_ROOT / output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "metadata": {
            "total_trips": total_trips,
            "successful_trips": successful_trips,
            "failed_trips": failed_trips,
            "total_wall_time_seconds": round(total_wall_time_s, 2),
            "mean_top1_accuracy_pct": round(float(np.mean(valid_top1)), 2) if valid_top1 else 0.0,
            "mean_route_continuity_pct": round(float(np.mean(valid_continuity)), 2) if valid_continuity else 0.0,
            "mean_candidate_recall_pct": round(float(np.mean(valid_recall)), 2) if valid_recall else 0.0,
            "mean_candidate_macro_f1": round(float(np.mean(valid_f1)), 4) if valid_f1 else 0.0,
            "mean_latency_ms": round(float(np.mean(valid_times)), 2) if valid_times else 0.0,
            "model_name": pipeline.model_name,
            "decoder": "HMM Viterbi",
        },
        "trips": results_list,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[+] Saved consolidated test results -> {output_json}")
    return failed_trips == 0


if __name__ == "__main__":
    success = run_all_trips_test()
    sys.exit(0 if success else 1)
