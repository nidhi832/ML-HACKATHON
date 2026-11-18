"""
PathSnap: Interactive Review & Demo CLI
For live demonstration in university hackathon review / evaluation panels.
Features:
  1. Displays multi-family benchmark comparison table.
  2. Runs instant inference & evaluation on any test trip.
  3. Displays detailed telemetry (Top-1 accuracy, candidate recall, route continuity).
  4. Automatically opens the interactive Folium HTML map in the default browser.
"""

import sys
import json
import webbrowser
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
MAPS_DIR = PROJECT_ROOT / "maps"


def print_banner():
    banner = r"""
================================================================================
   ____        __  __    ____                  
  / __ \____ _/ /_/ /_  / __/___  ____ _____   
 / /_/ / __ `/ __/ __ \/ /_/ __ \/ __ `/ __ \  
/ ____/ /_/ / /_/ / / / __/ / / / /_/ / /_/ /  
\/    \__,_/\__/_/ /_/_/ /_/ /_/\__,_/ .___/   
                                    /_/        
   Robust ML-Driven Map Matching System | PES University Hackathon
================================================================================
"""
    print(banner)


def show_benchmark_table():
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        print(f"[-] Benchmark results not found at {csv_path}. Please complete training first.")
        return
        
    df = pd.read_csv(csv_path)
    print("\n[+] 1. MULTI-FAMILY MODEL BENCHMARK (5-Fold GroupKFold by Trip):")
    print("-" * 105)
    header = f"{'Model Family':<28} | {'CV Macro F1':<15} | {'Test Top-1 Acc':<14} | {'Test Macro F1':<13} | {'Log Loss':<9} | {'Continuity'}"
    print(header)
    print("-" * 105)
    for _, r in df.sort_values(by="cv_macro_f1_mean", ascending=False).iterrows():
        f1_str = f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}"
        top1_acc = r['student_top1_acc']
        top1_str = f"{top1_acc:.2f}%" if top1_acc > 1.0 else f"{top1_acc*100:.2f}%"
        test_f1 = f"{r['student_macro_f1']:.4f}"
        ll_str = f"{r['log_loss']:.4f}"
        cont_val = r['route_continuity_pct']
        cont_str = f"{cont_val:.1f}%" if cont_val > 1.0 else f"{cont_val*100:.1f}%"
        print(f"{r['model_name']:<28} | {f1_str:<15} | {top1_str:<14} | {test_f1:<13} | {ll_str:<9} | {cont_str}")
    print("-" * 105)


def show_candidate_recall():
    miss_path = RESULTS_DIR / "candidate_miss_rate.csv"
    if miss_path.exists():
        miss_df = pd.read_csv(miss_path)
        print("\n[+] 2. CANDIDATE GENERATION RECALL (Spatial cKDTree Indexing):")
        print("-" * 65)
        for _, r in miss_df.iterrows():
            recall = r['recall_pct'] if 'recall_pct' in r else (1.0 - r['miss_rate']) * 100
            miss = r['miss_rate_pct'] if 'miss_rate_pct' in r else r['miss_rate'] * 100
            print(f"  Partition: {r['partition']:<14} | K = {int(r['K']):<2} | Recall: {recall:>6.2f}% | Miss: {miss:>5.2f}%")
        print("-" * 65)


def get_top1_predictions(df: pd.DataFrame) -> pd.DataFrame:
    prob_col = "prob_lightgbm" if "prob_lightgbm" in df.columns else (
        "prob_random_forest" if "prob_random_forest" in df.columns else df.columns[-1]
    )
    idx_max = df.groupby(["trip_id", "obs_idx"])[prob_col].idxmax()
    return df.loc[idx_max].copy(), prob_col


def show_trip_demo(trip_id: str = None):
    preds_file = RESULTS_DIR / "student_test_predictions.parquet"
    if not preds_file.exists():
        print("[-] Predictions file not generated yet.")
        return
        
    df = pd.read_parquet(preds_file)
    top1, prob_col = get_top1_predictions(df)
    trip_accs = top1.groupby("trip_id")["y"].mean().sort_values(ascending=False)
    
    if trip_id is None or trip_id not in trip_accs.index:
        trip_id = trip_accs.index[0]  # default to best
        
    acc = trip_accs[trip_id]
    n_pts = len(top1[top1["trip_id"] == trip_id])
    
    print(f"\n[+] 3. LIVE DEMO ON STUDENT-TEST TRIP: {trip_id}")
    print(f"  • GPS Observations: {n_pts:,} points")
    print(f"  • Top-1 Matching Accuracy ({prob_col}): {acc * 100:.2f}%")
    
    map_files = list(MAPS_DIR.glob(f"*{trip_id}*.html"))
    if map_files:
        map_path = map_files[0]
        print(f"  • Opening interactive multi-layer map in browser: {map_path.name}")
        try:
            webbrowser.open(map_path.resolve().as_uri())
        except Exception:
            print(f"  • Map saved at: {map_path}")
    else:
        print(f"  • Map file not found in {MAPS_DIR}. Run scripts/06_folium_maps.py.")


def main():
    print_banner()
    show_benchmark_table()
    show_candidate_recall()
    
    preds_file = RESULTS_DIR / "student_test_predictions.parquet"
    if preds_file.exists():
        df = pd.read_parquet(preds_file)
        top1, _ = get_top1_predictions(df)
        trip_accs = top1.groupby("trip_id")["y"].mean().sort_values(ascending=False)
        
        print("\nAvailable Student Test Trips:")
        for idx, (t_id, acc) in enumerate(trip_accs.items()):
            label = " (BEST)" if idx == 0 else (" (WORST)" if idx == len(trip_accs)-1 else "")
            print(f"  [{idx + 1}] Trip {t_id}: {acc*100:.2f}% accuracy{label}")
            
        chosen_trip = sys.argv[1] if len(sys.argv) > 1 else trip_accs.index[0]
        show_trip_demo(chosen_trip)
    else:
        print("\n[i] Run scripts/03_train_compare.py to enable live trip testing.")


if __name__ == "__main__":
    main()
