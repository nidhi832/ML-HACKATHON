"""
src/candidates.py - Fast candidate arc generation using spatial index and exact point-to-segment distance.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data_io import TripData
from src.geo import TripProjector, vectorised_point_to_segment


def generate_candidates_for_trip(
    trip: TripData,
    K: int = 10,
    midpoint_pool: int = 60,
    projector: Optional[TripProjector] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the K closest candidate arcs for each GPS observation in a trip.
    
    1. Projects GPS track and road nodes to a local metric UTM projection.
    2. Builds a cKDTree over road arc midpoints.
    3. Queries the top `midpoint_pool` (default 60) nearest arc midpoints per GPS point.
    4. Computes exact point-to-segment orthogonal distance and projection fraction.
    5. Sorts candidates by exact orthogonal distance and selects the top K closest arcs.
    
    Returns:
        candidates_df: Long table with columns:
            ['trip_id', 'obs_idx', 'arc_id', 'rank', 'dist_m', 'proj_frac']
        all_candidate_arcs_sorted: (N_gps, midpoint_pool) array of arc_ids ordered by exact distance,
            used for computing Candidate Miss Rate at various K.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    trip_id = trip.trip_id

    # 1. Metric UTM projection
    if projector is None:
        projector = TripProjector(track["lon"].mean(), track["lat"].mean())

    gx, gy = projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx, ny = projector.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())

    from_nds = arcs["from_node"].to_numpy()
    to_nds = arcs["to_node"].to_numpy()
    ax = nx[from_nds]
    ay = ny[from_nds]
    bx = nx[to_nds]
    by = ny[to_nds]

    # Arc midpoints
    mx = (ax + bx) * 0.5
    my = (ay + by) * 0.5

    # 2. Build spatial index over midpoints
    midpoints = np.column_stack([mx, my])
    tree = cKDTree(midpoints)

    # 3. Query top midpoint neighbours
    actual_pool = min(midpoint_pool, len(arcs))
    gps_pts = np.column_stack([gx, gy])
    _, pool_indices = tree.query(gps_pts, k=actual_pool, workers=-1)

    # 4. Compute exact point-to-segment distance for candidates
    N_pts = len(gx)
    cand_ax = ax[pool_indices]
    cand_ay = ay[pool_indices]
    cand_bx = bx[pool_indices]
    cand_by = by[pool_indices]

    dists, proj_fracs, _ = vectorised_point_to_segment(
        gx[:, None], gy[:, None],
        cand_ax, cand_ay,
        cand_bx, cand_by,
    )

    # 5. Sort by exact distance
    sorted_order = np.argsort(dists, axis=1)
    row_idx = np.arange(N_pts)[:, None]
    sorted_cand_arcs = pool_indices[row_idx, sorted_order]
    sorted_cand_dists = dists[row_idx, sorted_order]
    sorted_cand_fracs = proj_fracs[row_idx, sorted_order]

    # Keep top K
    actual_K = min(K, actual_pool)
    top_arcs = sorted_cand_arcs[:, :actual_K]
    top_dists = sorted_cand_dists[:, :actual_K]
    top_fracs = sorted_cand_fracs[:, :actual_K]

    # Flatten into long DataFrame
    obs_indices = np.repeat(np.arange(N_pts, dtype=np.int32), actual_K)
    ranks = np.tile(np.arange(actual_K, dtype=np.int16), N_pts)
    flat_arcs = top_arcs.flatten().astype(np.int32)
    flat_dists = top_dists.flatten().astype(np.float32)
    flat_fracs = top_fracs.flatten().astype(np.float32)

    df_cand = pd.DataFrame({
        "trip_id": trip_id,
        "obs_idx": obs_indices,
        "arc_id": flat_arcs,
        "rank": ranks,
        "dist_m": flat_dists,
        "proj_frac": flat_fracs,
    })

    return df_cand, sorted_cand_arcs
