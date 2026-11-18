"""
scripts/02_build_dataset.py - End-to-end Candidate Generation, Monotonic DP Labeling,
and Leak-Free Feature Matrix Construction for PathSnap.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io import load_trip, list_trips, save_parquet
from src.candidates import generate_candidates_for_trip
from src.labels import align_gps_to_route, generate_alignment_debug_map
from src.features import compute_candidate_features


def build_dataset():
    print("=" * 70)
    print("PathSnap: Building Feature Datasets & Candidate Labels...")
    print("=" * 70)

    os.makedirs("data_cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    partitions = ["train", "student_test"]
    k_eval_values = [5, 10, 15, 20]

    all_miss_rate_records = []
    alignment_records = []

    for part in partitions:
        trips = list_trips("Student folder", part)
        print(f"\nProcessing {part.upper()} partition ({len(trips)} trips)...")
        part_start_time = time.time()

        feature_dfs = []
        part_gps_count = 0
        part_hits = {k: 0 for k in k_eval_values}

        for idx, (tid, tdir) in enumerate(trips):
            t_trip_0 = time.time()
            trip = load_trip(tdir)
            N_pts = len(trip.track)
            part_gps_count += N_pts

            # 1. Candidate Generation (K=10, pool=60)
            cand_df, sorted_pool = generate_candidates_for_trip(trip, K=10, midpoint_pool=60)

            # 2. Ground-truth alignment via Monotonic DP
            align_df = align_gps_to_route(trip)
            aligned_arcs = align_df["aligned_arc_id"].to_numpy()

            # Record alignment quality
            align_dists = align_df["align_dist_m"].to_numpy()
            alignment_records.append({
                "partition": part,
                "trip_id": tid,
                "gps_count": N_pts,
                "median_align_dist_m": float(np.median(align_dists)),
                "p95_align_dist_m": float(np.percentile(align_dists, 95)),
                "pct_gt_50m": float(np.mean(align_dists > 50.0) * 100.0),
                "max_align_dist_m": float(np.max(align_dists)),
            })

            # Check candidate miss rate for K in {5, 10, 15, 20}
            for k in k_eval_values:
                in_top_k = np.any(sorted_pool[:, :k] == aligned_arcs[:, None], axis=1)
                part_hits[k] += int(np.sum(in_top_k))

            # Assign candidate ground-truth label y in {0, 1}
            aligned_map = align_df.set_index("obs_idx")["aligned_arc_id"].to_dict()
            cand_true_arc = cand_df["obs_idx"].map(aligned_map).to_numpy()
            cand_df["y"] = (cand_df["arc_id"].to_numpy() == cand_true_arc).astype(np.int8)

            # 3. Compute leak-free features
            feats_df = compute_candidate_features(trip, cand_df)
            feats_df["y"] = cand_df["y"].to_numpy()
            feature_dfs.append(feats_df)

            # Save debug maps for the first two training trips
            if part == "train" and tid in ["00000001", "00000002"]:
                debug_map_path = f"maps/alignment_debug_{tid}.html"
                generate_alignment_debug_map(trip, align_df, debug_map_path)
                print(f"  [Debug Map] Saved Folium alignment map -> {debug_map_path}")

            if (idx + 1) % 20 == 0 or (idx + 1) == len(trips):
                elapsed = time.time() - part_start_time
                print(f"  Processed {idx + 1}/{len(trips)} trips ({round(elapsed, 1)}s)...")

        # Concatenate and save partition parquet
        combined_feats = pd.concat(feature_dfs, ignore_index=True)
        parquet_path = f"data_cache/{part}_features.parquet"
        save_parquet(combined_feats, parquet_path)
        file_sz_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  Saved {parquet_path} ({file_sz_mb:.1f} MB, {len(combined_feats):,} candidate rows)")

        # Record miss rates for partition
        for k in k_eval_values:
            recall = part_hits[k] / max(1, part_gps_count)
            miss_rate = 1.0 - recall
            all_miss_rate_records.append({
                "partition": part,
                "K": k,
                "total_observations": part_gps_count,
                "hits": part_hits[k],
                "recall_pct": round(recall * 100.0, 2),
                "miss_rate_pct": round(miss_rate * 100.0, 2),
            })

    # Save candidate miss rate table
    df_miss = pd.DataFrame(all_miss_rate_records)
    df_miss.to_csv("results/candidate_miss_rate.csv", index=False)
    print("\nCandidate Miss Rate Summary (saved to results/candidate_miss_rate.csv):")
    print(df_miss.to_string(index=False))

    # Save alignment quality table
    df_align = pd.DataFrame(alignment_records)
    df_align.to_csv("results/alignment_quality.csv", index=False)
    print("\nAlignment Quality Summary:")
    print(f"  Overall Median Distance: {df_align['median_align_dist_m'].mean():.2f} m")
    print(f"  Overall 95th Percentile: {df_align['p95_align_dist_m'].mean():.2f} m")
    print(f"  Observations > 50m: {df_align['pct_gt_50m'].mean():.2f}%")

    # Load and print feature & class summary
    train_df = pd.read_parquet("data_cache/train_features.parquet")
    test_df = pd.read_parquet("data_cache/student_test_features.parquet")

    print("\n" + "=" * 70)
    print("Class Balance & Dataset Summary:")
    print("=" * 70)
    for name, df in [("TRAIN", train_df), ("STUDENT_TEST", test_df)]:
        n_pos = int((df["y"] == 1).sum())
        n_neg = int((df["y"] == 0).sum())
        total = len(df)
        print(f"{name:12s} Total: {total:,} rows | Positive (y=1): {n_pos:,} ({n_pos/total*100:.2f}%) | Negative (y=0): {n_neg:,} ({n_neg/total*100:.2f}%)")

    # Write feature_list.md
    print("\nWriting feature_list.md documentation...")
    feature_doc = """# PathSnap Feature Set Documentation & Leakage Audit

Every feature in the candidate ranking model is computed strictly using only the raw GPS trajectory (`.track`) and the road network graph topology (`.nodes`, `.arcs`). **No feature accesses or depends upon `.route` or ground-truth labels.**

## 1. Candidate Geometry & Proximity Features
- `dist_m`: Shortest perpendicular / orthogonal distance from the GPS point to the candidate road segment (in metres). Primary geometric indicator of proximity.
- `proj_frac`: Clamped projection fraction $\in [0.0, 1.0]$ indicating whether the observation projects near the start (0), middle, or end (1) of the road segment.
- `raw_proj_frac`: Unclamped projection scalar $t$; values $<0$ or $>1$ identify observations projecting beyond the segment endpoints.
- `arc_length_m`: Physical length of the candidate road arc in metres. Longer arcs naturally accommodate more consecutive GPS observations.
- `candidate_rank`: 0-indexed distance rank among the $K=10$ candidate arcs (0 = closest segment).
- `dist_gap_to_best_m`: Distance delta between this candidate's distance and the rank-0 candidate's distance (`dist_m - min_dist_m`). Captures candidate ambiguity.
- `n_cand_within_10m`: Number of candidate road arcs within 10 metres of the observation. Measures local road density and junction complexity.
- `n_cand_within_25m`: Number of candidate road arcs within 25 metres of the observation.
- `n_cand_within_50m`: Number of candidate road arcs within 50 metres of the observation.

## 2. Heading, Kinematics & Directional Alignment
- `arc_bearing`: Compass bearing $[0, 360)^\circ$ along the directed road arc from `from_node` to `to_node`.
- `gps_heading`: Smoothed trajectory heading $[0, 360)^\circ$ computed from a symmetric window ($\pm 2$ points) to filter sensor jitter.
- `heading_abs_diff`: Absolute angular difference $| \text{gps\_heading} - \text{arc\_bearing} | \in [0, 180]^\circ$. Critical for eliminating opposing one-way or dual-carriageway segments.
- `heading_cos_diff`: Cosine of the angular difference $\cos(\text{gps\_heading} - \text{arc\_bearing}) \in [-1.0, 1.0]$. Directly aligns forward vehicle travel ($+1$) against wrong-way travel ($-1$).
- `speed_mps`: Instantaneous vehicle travel speed in metres per second.
- `is_stationary`: Binary indicator (1 if speed $< 0.5$ m/s, else 0); suppresses unreliable heading changes during traffic stops or red lights.
- `turn_angle_deg`: Heading change magnitude from the previous observation in degrees $[0, 180]^\circ$. Flags vehicle turns and intersection maneuvers.
- `acceleration_mps2`: Vehicle acceleration in $\text{m/s}^2$ computed over consecutive observations.
- `dt_prev`: Sampling time interval from the previous GPS observation (seconds).
- `dt_next`: Sampling time interval to the next GPS observation (seconds).

## 3. Road Network Graph Topology & Junction Structure
- `arc_out_degree`: Number of outgoing directed road arcs from the candidate arc's destination node (`to_node`). Represents upcoming routing choices.
- `arc_in_degree`: Number of incoming directed road arcs into the candidate arc's origin node (`from_node`).
- `has_reverse_arc`: Binary flag (1 if an opposing directed edge exists between the same node pair, 0 otherwise). Crucial for disambiguating two-way streets versus dual carriageways.

## 4. Spatiotemporal Trajectory Context
- `prev_point_dist_to_arc`: Orthogonal distance from the previous GPS observation $P_{i-1}$ to this candidate arc segment.
- `next_point_dist_to_arc`: Orthogonal distance from the next GPS observation $P_{i+1}$ to this candidate arc segment.
- `rolling_dist_mean_3`: 3-point rolling mean distance $(d_{i-1} + d_i + d_{i+1}) / 3$ to this segment, smoothing temporary GPS multipath excursions.
- `dist_change_prev`: Rate of distance change relative to the segment ($d_i - d_{i-1}$); negative values indicate approaching the segment, positive indicate leaving.

## 5. Leakage Verification Checklist
- [x] Absolute latitude and longitude coordinates are excluded from model features.
- [x] Global trip identifiers, timestamps, and row indices are excluded from feature vectors.
- [x] The reference route sequence (`.route`) is NEVER accessed during candidate generation or feature computation.
- [x] Scalers and preprocessors are strictly fit on the training partition and applied out-of-sample to validation/testing sets.
"""
    with open("feature_list.md", "w") as f:
        f.write(feature_doc)

    print("feature_list.md successfully written.")
    print("=" * 70)
    print("Dataset build complete!")
    print("=" * 70)


if __name__ == "__main__":
    build_dataset()
