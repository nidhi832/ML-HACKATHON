"""
scripts/pre_evaluation_check.py - Automated Pre-Evaluation Checklist for PathSnap & TRACE.AI.

Runs end-to-end verification of:
1. GET /health: Verifies API status, model loading, HMM decoder, and dataset count.
2. GET /trips: Verifies trip directory enumeration across student_test partition.
3. POST /predict: Evaluates all 10 student-test trips live via FastAPI endpoints.
4. Checks HMM vs Classifier Argmax decoder switching and continuity.
5. Prints a structured PASS/FAIL terminal summary table.

Can run against a live running server:
    python scripts/pre_evaluation_check.py --url http://127.0.0.1:8000
Or run directly in-process:
    python scripts/pre_evaluation_check.py
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Prevent joblib loky from calling external subprocess to count CPUs in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import (
    app,
    health_check,
    get_trips,
    predict_trip_route,
    PredictRequest,
    EVAL_DATA_DIR,
)


class DirectApiClient:
    """Invokes FastAPI handler functions directly without network/subprocess overhead."""
    def get_health(self) -> Dict[str, Any]:
        resp = health_check()
        return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()

    def get_trips(self, partition: str = "student_test") -> Dict[str, Any]:
        resp = get_trips(partition=partition)
        return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()

    def post_predict(self, body: dict) -> Dict[str, Any]:
        req = PredictRequest(**body)
        resp = predict_trip_route(req)
        return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()


class RemoteApiClient:
    """Interacts with remote running FastAPI server via requests."""
    def __init__(self, base_url: str):
        import requests
        self.requests = requests
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        res = self.requests.get(f"{self.base_url}/health", timeout=10)
        res.raise_for_status()
        return res.json()

    def get_trips(self, partition: str = "student_test") -> Dict[str, Any]:
        res = self.requests.get(f"{self.base_url}/trips", params={"partition": partition}, timeout=10)
        res.raise_for_status()
        return res.json()

    def post_predict(self, body: dict) -> Dict[str, Any]:
        res = self.requests.post(f"{self.base_url}/predict", json=body, timeout=20)
        res.raise_for_status()
        return res.json()


def run_checklist(base_url: str = None) -> bool:
    if base_url:
        client = RemoteApiClient(base_url)
        mode_desc = f"Remote URL: {base_url}"
    else:
        client = DirectApiClient()
        mode_desc = "Direct In-Process (Core Engine)"

    print("=" * 115)
    print("                      TRACE.AI & PathSnap Pre-Evaluation Readiness Checklist")
    print(f" Mode: {mode_desc}")
    print("=" * 115)

    all_passed = True

    # 1. Health Check
    print("\n[Step 1/3] Verifying /health endpoint...")
    t0 = time.time()
    try:
        h_data = client.get_health()
        health_time = round((time.time() - t0) * 1000, 1)
        model_ok = h_data.get("model_loaded") is True
        hmm_ok = h_data.get("hmm_decoder_loaded") is True
        trips_cnt = h_data.get("student_test_trips_count", 0)

        print(f"  -> Model Status: {'LOADED' if model_ok else 'FAILED'} ({h_data.get('model_name')})")
        print(f"  -> HMM Decoder:  {'LOADED' if hmm_ok else 'FAILED'}")
        print(f"  -> Active Trips: {trips_cnt} discovered ({health_time}ms)")

        if not (model_ok and hmm_ok and trips_cnt >= 10):
            all_passed = False
            print("  [FAIL] Health check returned incomplete system components.")
        else:
            print("  [PASS] System core engine and decoders verified.")
    except Exception as e:
        print(f"  [FAIL] Failed to connect to /health: {e}")
        return False

    # 2. Trips Listing
    print("\n[Step 2/3] Verifying /trips endpoint...")
    try:
        t_data = client.get_trips(partition="student_test")
        trip_ids = t_data.get("trip_ids", [])
        total_trips = t_data.get("total_trips", 0)
        print(f"  -> Discovered {total_trips} trips in partition 'student_test':")
        print(f"     {', '.join(trip_ids)}")
        if total_trips < 10:
            print(f"  [WARNING] Expected at least 10 trips, found {total_trips}")
        else:
            print("  [PASS] Trip enumeration validated.")
    except Exception as e:
        print(f"  [FAIL] Error querying /trips: {e}")
        all_passed = False
        trip_ids = []

    if not trip_ids:
        trip_ids = ["00000000", "00000007", "00000009", "00000040", "00000052",
                    "00000056", "00000065", "00000076", "00000077", "00000091"]

    # 3. Predict All Trips
    print("\n[Step 3/3] Executing Live Predictions Across All 10 Trips...")
    print("-" * 115)
    header = (
        f"{'Trip ID':<10} | {'Status':<6} | {'Points':<6} | {'Traversed':<9} | "
        f"{'Top-1 Acc':<9} | {'Continuity':<10} | {'Route F1':<8} | {'Latency':<8} | {'Decoder':<14}"
    )
    print(header)
    print("-" * 115)

    results = []
    for tid in trip_ids:
        try:
            req_body = {
                "trip_id": tid,
                "partition": "student_test",
                "model_name": "lightgbm",
                "use_hmm": True
            }
            t_pred_start = time.time()
            p_data = client.post_predict(req_body)
            p_time = round((time.time() - t_pred_start) * 1000, 1)

            n_obs = p_data.get("num_observations", 0)
            n_trav = p_data.get("num_arcs_traversed", 0)
            pred_arcs = p_data.get("predicted_arcs", [])
            cont = p_data.get("continuity_pct", 0.0)
            top1 = p_data.get("top1_accuracy_pct")
            f1 = p_data.get("route_lcs_f1")
            decoder = p_data.get("decoder", "HMM Viterbi")

            # Validation checks
            is_valid = (
                len(pred_arcs) == n_obs and
                n_obs > 0 and
                cont >= 95.0 and
                decoder == "HMM Viterbi"
            )
            status_str = "PASS" if is_valid else "WARN"
            if not is_valid:
                all_passed = False

            top1_str = f"{top1:.2f}%" if top1 is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"

            print(
                f"{tid:<10} | {status_str:<6} | {n_obs:<6} | {n_trav:<9} | "
                f"{top1_str:>9} | {cont:>9.2f}% | {f1_str:>8} | {p_time:>6.1f}ms | {decoder:<14}"
            )
            results.append({
                "trip_id": tid, "status": status_str, "obs": n_obs,
                "continuity": cont, "top1": top1, "f1": f1, "time": p_time
            })
        except Exception as exc:
            print(f"{tid:<10} | ERROR  | Exception: {exc}")
            all_passed = False

    # 4. Test Classifier Argmax (use_hmm=False)
    print("\n[*] Testing Decoder Switching (use_hmm=False -> Classifier Argmax)...")
    try:
        d_no_hmm = client.post_predict({
            "trip_id": "00000052",
            "partition": "student_test",
            "model_name": "lightgbm",
            "use_hmm": False
        })
        dec_name = d_no_hmm.get("decoder")
        cont_no_hmm = d_no_hmm.get("continuity_pct")
        print(f"  -> Decoder: {dec_name} (Continuity: {cont_no_hmm:.2f}%)")
        if dec_name == "Classifier Argmax":
            print("  [PASS] Live decoder toggle verified.")
        else:
            print(f"  [FAIL] Expected 'Classifier Argmax', got '{dec_name}'")
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] Exception during decoder toggle check: {e}")
        all_passed = False

    # Summary
    print("=" * 115)
    if results:
        mean_cont = sum(r["continuity"] for r in results) / len(results)
        valid_top1s = [r["top1"] for r in results if r["top1"] is not None]
        mean_top1 = sum(valid_top1s) / len(valid_top1s) if valid_top1s else 0.0
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"Summary Statistics:")
        print(f"  - Total Trips Evaluated : {len(results)}/10")
        print(f"  - Mean Route Continuity : {mean_cont:.2f}%")
        print(f"  - Mean Top-1 Accuracy   : {mean_top1:.2f}%")
        print(f"  - Average Trip Latency  : {avg_time:.1f}ms")

    if all_passed:
        print("\n>>> OVERALL RESULT: ALL CHECKS PASSED - READY FOR HACKATHON EVALUATION <<<\n")
    else:
        print("\n>>> OVERALL RESULT: SOME CHECKS REQUIRE ATTENTION <<<\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE.AI Pre-Evaluation Checklist")
    parser.add_argument("--url", type=str, default=None, help="Base API URL (e.g. http://127.0.0.1:8000)")
    args = parser.parse_args()

    success = run_checklist(base_url=args.url)
    sys.exit(0 if success else 1)
