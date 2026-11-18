"""
src/features.py - Feature engineering for candidate road arcs.
All features are derived strictly from GPS observations and road network topology,
ensuring ZERO leakage of ground-truth reference routes.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.data_io import TripData
from src.geo import (
    TripProjector,
    vectorised_point_to_segment,
    calculate_bearing,
    calculate_gps_headings,
    angle_difference,
)


def compute_trip_kinematics(
    gx: np.ndarray,
    gy: np.ndarray,
    t: np.ndarray,
) -> dict:
    """
    Compute kinematic trajectory features for all GPS points in a trip:
    - Speed (m/s)
    - Acceleration (m/s^2)
    - Turn angles (deg)
    - Temporal intervals dt_prev, dt_next
    - Smoothed heading
    - Stationary flag
    """
    N = len(gx)
    
    # Time deltas
    dt_prev = np.zeros(N, dtype=np.float32)
    dt_next = np.zeros(N, dtype=np.float32)
    if N > 1:
        dt = np.diff(t).astype(np.float32)
        dt_safe = np.maximum(dt, 1e-3)
        dt_prev[1:] = dt_safe
        dt_prev[0] = dt_safe[0]
        dt_next[:-1] = dt_safe
        dt_next[-1] = dt_safe[-1]
    else:
        dt_prev[:] = 1.0
        dt_next[:] = 1.0

    # Step distances
    step_dist = np.zeros(N, dtype=np.float32)
    if N > 1:
        dx = np.diff(gx)
        dy = np.diff(gy)
        d = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        step_dist[1:] = d
        step_dist[0] = d[0]

    # Speeds (m/s) using symmetric differences where possible
    speed = np.zeros(N, dtype=np.float32)
    for i in range(N):
        i_prev = max(0, i - 1)
        i_next = min(N - 1, i + 1)
        dt_span = max(1e-3, t[i_next] - t[i_prev])
        dist_span = np.sqrt((gx[i_next] - gx[i_prev])**2 + (gy[i_next] - gy[i_prev])**2)
        speed[i] = dist_span / dt_span

    # Accelerations (m/s^2)
    accel = np.zeros(N, dtype=np.float32)
    for i in range(N):
        i_prev = max(0, i - 1)
        i_next = min(N - 1, i + 1)
        dt_span = max(1e-3, t[i_next] - t[i_prev])
        accel[i] = (speed[i_next] - speed[i_prev]) / dt_span

    # Smoothed headings
    gps_headings = calculate_gps_headings(gx, gy, window=2).astype(np.float32)

    # Turn angles (absolute heading changes)
    turn_angles = np.zeros(N, dtype=np.float32)
    if N > 1:
        diff_h = (gps_headings[1:] - gps_headings[:-1] + 180.0) % 360.0 - 180.0
        turn_angles[1:] = np.abs(diff_h)
        turn_angles[0] = turn_angles[1] if N > 1 else 0.0

    is_stationary = (speed < 0.5).astype(np.float32)

    return {
        "gps_headings": gps_headings,
        "speed": speed,
        "accel": accel,
        "turn_angles": turn_angles,
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "is_stationary": is_stationary,
    }


def compute_candidate_features(
    trip: TripData,
    cand_df: pd.DataFrame,
    projector: Optional[TripProjector] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive features for each candidate row in cand_df.
    
    Inputs:
        trip: TripData namedtuple (track, nodes, arcs, route, trip_id)
        cand_df: Candidate DataFrame ['trip_id', 'obs_idx', 'arc_id', 'rank', 'dist_m', 'proj_frac']
        projector: Optional TripProjector
        
    Returns:
        Augmented DataFrame with all predictive features ready for model training/inference.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    N_pts = len(track)
    N_nodes = len(nodes)
    N_arcs = len(arcs)

    # 1. Metric coordinates
    if projector is None:
        projector = TripProjector(track["lon"].mean(), track["lat"].mean())

    gx, gy = projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx, ny = projector.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    t = track["t"].to_numpy()

    # 2. Kinematics for GPS track
    kinematics = compute_trip_kinematics(gx, gy, t)

    # 3. Network graph properties (degrees, reverse arcs)
    from_nds = arcs["from_node"].to_numpy()
    to_nds = arcs["to_node"].to_numpy()

    out_degree = np.bincount(from_nds, minlength=N_nodes).astype(np.int16)
    in_degree = np.bincount(to_nds, minlength=N_nodes).astype(np.int16)

    # Fast reverse arc existence check via 64-bit hashed edge keys
    edge_keys = (from_nds.astype(np.int64) << 32) | to_nds.astype(np.int64)
    edge_set = set(edge_keys)
    rev_keys = (to_nds.astype(np.int64) << 32) | from_nds.astype(np.int64)
    has_reverse = np.array([k in edge_set for k in rev_keys], dtype=np.int8)

    # Arc geometries
    ax = nx[from_nds]
    ay = ny[from_nds]
    bx = nx[to_nds]
    by = ny[to_nds]
    arc_lengths = np.sqrt((bx - ax)**2 + (by - ay)**2).astype(np.float32)
    arc_bearings = calculate_bearing(ax, ay, bx, by).astype(np.float32)

    # 4. Map candidate rows
    obs_idx = cand_df["obs_idx"].to_numpy()
    cand_arc_id = cand_df["arc_id"].to_numpy()
    cand_dist_m = cand_df["dist_m"].to_numpy()
    cand_proj_frac = cand_df["proj_frac"].to_numpy()
    cand_rank = cand_df["rank"].to_numpy()

    # Candidate arc coordinates
    c_ax = ax[cand_arc_id]
    c_ay = ay[cand_arc_id]
    c_bx = bx[cand_arc_id]
    c_by = by[cand_arc_id]
    c_arc_len = arc_lengths[cand_arc_id]
    c_arc_bearing = arc_bearings[cand_arc_id]

    # Heading features
    c_gps_heading = kinematics["gps_headings"][obs_idx]
    head_abs_diff, head_cos_diff = angle_difference(c_gps_heading, c_arc_bearing)

    # Raw projection fraction
    _, _, raw_frac = vectorised_point_to_segment(
        gx[obs_idx], gy[obs_idx],
        c_ax, c_ay,
        c_bx, c_by
    )

    # Observation-level aggregation: gap to best candidate & candidate density
    K = int(cand_df.groupby("obs_idx")["rank"].count().iloc[0])
    num_obs = len(gx)

    # Reshape distances into (num_obs, K)
    dist_matrix = cand_dist_m.reshape(num_obs, K)
    min_dists = dist_matrix[:, 0]  # rank 0 is closest
    dist_gap = cand_dist_m - np.repeat(min_dists, K)

    n_within_10 = np.sum(dist_matrix <= 10.0, axis=1)
    n_within_25 = np.sum(dist_matrix <= 25.0, axis=1)
    n_within_50 = np.sum(dist_matrix <= 50.0, axis=1)

    c_n_within_10 = np.repeat(n_within_10, K)
    c_n_within_25 = np.repeat(n_within_25, K)
    c_n_within_50 = np.repeat(n_within_50, K)

    # Temporal context: distance from neighbouring GPS points (obs_idx - 1, obs_idx + 1) to candidate arc
    prev_obs_idx = np.maximum(0, obs_idx - 1)
    next_obs_idx = np.minimum(num_obs - 1, obs_idx + 1)

    prev_dist, _, _ = vectorised_point_to_segment(
        gx[prev_obs_idx], gy[prev_obs_idx],
        c_ax, c_ay,
        c_bx, c_by
    )
    next_dist, _, _ = vectorised_point_to_segment(
        gx[next_obs_idx], gy[next_obs_idx],
        c_ax, c_ay,
        c_bx, c_by
    )

    rolling_dist_mean_3 = (prev_dist + cand_dist_m + next_dist) / 3.0
    dist_change_prev = cand_dist_m - prev_dist

    # Network topology features for candidate arc
    cand_out_deg = out_degree[to_nds[cand_arc_id]]
    cand_in_deg = in_degree[from_nds[cand_arc_id]]
    cand_has_rev = has_reverse[cand_arc_id]

    # Build features dictionary
    features = {
        "trip_id": cand_df["trip_id"].to_numpy(),
        "obs_idx": obs_idx,
        "arc_id": cand_arc_id,
        "dist_m": cand_dist_m,
        "proj_frac": cand_proj_frac,
        "raw_proj_frac": raw_frac.astype(np.float32),
        "arc_length_m": c_arc_len,
        "candidate_rank": cand_rank.astype(np.int16),
        "dist_gap_to_best_m": dist_gap.astype(np.float32),
        "n_cand_within_10m": c_n_within_10.astype(np.int16),
        "n_cand_within_25m": c_n_within_25.astype(np.int16),
        "n_cand_within_50m": c_n_within_50.astype(np.int16),
        "arc_bearing": c_arc_bearing,
        "gps_heading": c_gps_heading,
        "heading_abs_diff": head_abs_diff.astype(np.float32),
        "heading_cos_diff": head_cos_diff.astype(np.float32),
        "speed_mps": kinematics["speed"][obs_idx],
        "is_stationary": kinematics["is_stationary"][obs_idx],
        "turn_angle_deg": kinematics["turn_angles"][obs_idx],
        "acceleration_mps2": kinematics["accel"][obs_idx],
        "dt_prev": kinematics["dt_prev"][obs_idx],
        "dt_next": kinematics["dt_next"][obs_idx],
        "prev_point_dist_to_arc": prev_dist.astype(np.float32),
        "next_point_dist_to_arc": next_dist.astype(np.float32),
        "rolling_dist_mean_3": rolling_dist_mean_3.astype(np.float32),
        "dist_change_prev": dist_change_prev.astype(np.float32),
        "arc_out_degree": cand_out_deg,
        "arc_in_degree": cand_in_deg,
        "has_reverse_arc": cand_has_rev,
    }

    return pd.DataFrame(features)
