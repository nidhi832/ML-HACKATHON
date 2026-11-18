"""
src/labels.py - Monotonic dynamic programming alignment of GPS points to reference route.
Generates ground-truth candidate labels (y in {0, 1}) for training and validation.
"""

from typing import Optional, Tuple, Dict, List
import os
import numpy as np
import pandas as pd
import folium

from src.data_io import TripData
from src.geo import (
    TripProjector,
    vectorised_point_to_segment,
    calculate_bearing,
    calculate_gps_headings,
    angle_difference,
)


def align_gps_to_route(
    trip: TripData,
    projector: Optional[TripProjector] = None,
    heading_weight: float = 12.0,
    max_hops: int = 3,
) -> pd.DataFrame:
    """
    Monotonic alignment of GPS points to reference route arc sequence by dynamic programming.
    
    State: route index j (0 <= j < M), where route[j] is the traversed arc_id.
    Cost: point-to-arc distance (m) + heading mismatch penalty.
    Transitions: j -> j, j -> j+1, j -> j+2, j -> j+3 (monotonic, forward only).
    
    Returns:
        DataFrame with columns ['trip_id', 'obs_idx', 'aligned_route_idx', 'aligned_arc_id', 'align_dist_m']
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route data.")

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    T = len(track)
    M = len(route)

    # 1. Project to local metric UTM
    if projector is None:
        projector = TripProjector(track["lon"].mean(), track["lat"].mean())

    gx, gy = projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx, ny = projector.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())

    # GPS headings
    gps_headings = calculate_gps_headings(gx, gy, window=2)

    # Route arc segments
    from_nds = arcs["from_node"].to_numpy()[route]
    to_nds = arcs["to_node"].to_numpy()[route]
    r_ax = nx[from_nds]
    r_ay = ny[from_nds]
    r_bx = nx[to_nds]
    r_by = ny[to_nds]

    route_bearings = calculate_bearing(r_ax, r_ay, r_bx, r_by)

    # Heading difference penalty: heading_weight * (1 - cos(diff))
    _, cos_diff = angle_difference(gps_headings[:, None], route_bearings[None, :])
    heading_pen = heading_weight * (1.0 - cos_diff)

    # Fast distance computation (chunked to ensure strictly bounded RAM)
    chunk_sz = 1500
    cost_matrix = np.empty((T, M), dtype=np.float32)
    dist_matrix = np.empty((T, M), dtype=np.float32)

    for c_start in range(0, T, chunk_sz):
        c_end = min(T, c_start + chunk_sz)
        dists_c, _, _ = vectorised_point_to_segment(
            gx[c_start:c_end, None], gy[c_start:c_end, None],
            r_ax[None, :], r_ay[None, :],
            r_bx[None, :], r_by[None, :]
        )
        dist_matrix[c_start:c_end] = dists_c.astype(np.float32)
        cost_matrix[c_start:c_end] = (dists_c + heading_pen[c_start:c_end]).astype(np.float32)

    # Dynamic Programming
    V = np.full(M, 1e9, dtype=np.float32)
    # Vehicle starts near the beginning of the route
    start_window = min(8, M)
    V[:start_window] = cost_matrix[0, :start_window]

    backpointers = np.zeros((T, M), dtype=np.int8)
    penalties = np.array([0.0, 1.5, 5.0, 15.0], dtype=np.float32)

    for i in range(1, T):
        c = cost_matrix[i]
        c0 = V + penalties[0]
        c1 = np.empty_like(V)
        c1[0] = 1e9
        c1[1:] = V[:-1] + penalties[1]

        c2 = np.empty_like(V)
        c2[:2] = 1e9
        c2[2:] = V[:-2] + penalties[2]

        c3 = np.empty_like(V)
        c3[:3] = 1e9
        c3[3:] = V[:-3] + penalties[3]

        stacked = np.stack([c0, c1, c2, c3], axis=0)
        best_trans = np.argmin(stacked, axis=0).astype(np.int8)
        backpointers[i] = best_trans
        V = np.min(stacked, axis=0) + c

    # Backtrack
    aligned_route_idx = np.zeros(T, dtype=np.int32)
    # End at the lowest cost state in the last portion of route
    end_window_start = max(0, M - min(10, M))
    aligned_route_idx[-1] = end_window_start + np.argmin(V[end_window_start:])

    for i in range(T - 1, 0, -1):
        step = backpointers[i, aligned_route_idx[i]]
        aligned_route_idx[i - 1] = aligned_route_idx[i] - step

    aligned_arc_id = route[aligned_route_idx]
    row_indices = np.arange(T)
    aligned_dists = dist_matrix[row_indices, aligned_route_idx]

    df_aligned = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": np.arange(T, dtype=np.int32),
        "aligned_route_idx": aligned_route_idx,
        "aligned_arc_id": aligned_arc_id,
        "align_dist_m": aligned_dists,
    })

    return df_aligned


def generate_alignment_debug_map(
    trip: TripData,
    aligned_df: pd.DataFrame,
    out_html: str,
    max_pts: int = 1500,
):
    """
    Generate an interactive Folium map visualizing the DP alignment quality:
    - Raw GPS points colored by alignment distance (Green <=10m, Orange 10-30m, Red >30m)
    - Aligned route polyline
    - Nearby road network
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri WorldStreetMap (Clear Roads)",
        overlay=False,
        control=True
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; CARTO",
        name="CartoDB Light",
        overlay=False,
        control=True
    ).add_to(fmap)

    nodes_arr = nodes[["lon", "lat"]].to_numpy()
    from_nds = arcs["from_node"].to_numpy()
    to_nds = arcs["to_node"].to_numpy()

    # Road network layer
    fg_net = folium.FeatureGroup(name="Road Network (Grey)", show=False)
    min_lat, max_lat = track["lat"].min() - 0.01, track["lat"].max() + 0.01
    min_lon, max_lon = track["lon"].min() - 0.01, track["lon"].max() + 0.01
    mask = (
        (nodes_arr[from_nds, 0] >= min_lon) & (nodes_arr[from_nds, 0] <= max_lon) &
        (nodes_arr[from_nds, 1] >= min_lat) & (nodes_arr[from_nds, 1] <= max_lat)
    )
    s_from = from_nds[mask]
    s_to = to_nds[mask]
    step = max(1, len(s_from) // 2500)
    for u, v in zip(s_from[::step], s_to[::step]):
        folium.PolyLine(
            [[nodes_arr[u, 1], nodes_arr[u, 0]], [nodes_arr[v, 1], nodes_arr[v, 0]]],
            color="#999999", weight=1.2, opacity=0.5
        ).add_to(fg_net)
    fg_net.add_to(fmap)

    # Reference route layer
    fg_route = folium.FeatureGroup(name="Reference Route (Green)", show=True)
    if route is not None:
        r_from = from_nds[route]
        r_to = to_nds[route]
        for u, v in zip(r_from, r_to):
            folium.PolyLine(
                [[nodes_arr[u, 1], nodes_arr[u, 0]], [nodes_arr[v, 1], nodes_arr[v, 0]]],
                color="#2ca02c", weight=3.5, opacity=0.85
            ).add_to(fg_route)
    fg_route.add_to(fmap)

    # Aligned GPS points layer
    fg_gps = folium.FeatureGroup(name="Aligned GPS Points", show=True)
    sub_df = aligned_df.iloc[::max(1, len(aligned_df) // max_pts)]
    for _, row in sub_df.iterrows():
        i = int(row["obs_idx"])
        d = float(row["align_dist_m"])
        lat = track.iloc[i]["lat"]
        lon = track.iloc[i]["lon"]
        color = "#2ca02c" if d <= 10.0 else ("#ff7f0e" if d <= 30.0 else "#d62728")
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Obs {i} | Arc {int(row['aligned_arc_id'])} | Dist: {d:.1f}m",
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.save(out_html)
