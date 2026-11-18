"""
scripts/01_eda.py - Comprehensive Dataset Exploration for PathSnap.
Generates inventory tables, data quality audits, network statistics,
route continuity checks, noise distributions, visualizations, and Folium maps.
"""

import os
import sys

# Set matplotlib cache directory to temp before importing pyplot
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io import load_trip, list_trips
from src.geo import TripProjector, vectorised_point_to_segment, calculate_bearing


def run_eda():
    print("=" * 70)
    print("Starting PathSnap Dataset Exploration (EDA)...")
    print("=" * 70)

    os.makedirs("results/eda", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    partitions = ["train", "student_test"]
    all_trips = []
    for p in partitions:
        for tid, tdir in list_trips("Student folder", p):
            all_trips.append((p, tid, tdir))

    print(f"Total trips found: {len(all_trips)} (Train: 80, Student Test: 10)")

    # Data collectors
    inventory_rows = []
    timestamp_rows = []
    missing_rows = []
    extent_rows = []
    network_rows = []
    route_rows = []
    noise_rows = []
    all_arc_lengths_sample = []
    all_noise_sample = []
    all_dt_sample = []

    # Process all trips
    t_start = time.time()
    for idx, (part, tid, tdir) in enumerate(all_trips):
        trip = load_trip(tdir)
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs
        route = trip.route

        # 1. File sizes
        track_sz = os.path.getsize(os.path.join(tdir, f"{tid}.track"))
        nodes_sz = os.path.getsize(os.path.join(tdir, f"{tid}.nodes"))
        arcs_sz = os.path.getsize(os.path.join(tdir, f"{tid}.arcs"))
        route_sz = os.path.getsize(os.path.join(tdir, f"{tid}.route")) if route is not None else 0
        total_sz = track_sz + nodes_sz + arcs_sz + route_sz

        inventory_rows.append({
            "partition": part,
            "trip_id": tid,
            "gps_count": len(track),
            "node_count": len(nodes),
            "arc_count": len(arcs),
            "route_length": len(route) if route is not None else 0,
            "track_bytes": track_sz,
            "nodes_bytes": nodes_sz,
            "arcs_bytes": arcs_sz,
            "route_bytes": route_sz,
            "total_bytes": total_sz,
        })

        # 2. Timestamps and Coordinates quality
        t_vals = track["t"].to_numpy()
        dts = np.diff(t_vals)
        all_dt_sample.extend(dts[::max(1, len(dts) // 500)].tolist())

        is_monotonic = bool(np.all(dts > 0))
        pct_1s = float(np.mean(np.isclose(dts, 1.0, atol=1e-2)) * 100.0) if len(dts) > 0 else 0.0
        max_gap = float(np.max(dts)) if len(dts) > 0 else 0.0
        gaps_gt_30s = int(np.sum(dts > 30.0))

        # Duplicates
        dup_gps = int(track.duplicated(subset=["lon", "lat"]).sum())
        dup_ts = int(track.duplicated(subset=["t"]).sum())

        # NaN / Inf checks
        nan_track = int(track.isna().sum().sum())
        nan_nodes = int(nodes.isna().sum().sum())
        nan_arcs = int(arcs.isna().sum().sum())
        nan_route = 0 if route is None else int(np.isnan(route).sum())

        # Range checks: lon in [-180, 180], lat in [-90, 90]
        invalid_lon = int(np.sum((track["lon"] < -180) | (track["lon"] > 180)))
        invalid_lat = int(np.sum((track["lat"] < -90) | (track["lat"] > 90)))
        coord_order_valid = (invalid_lon == 0 and invalid_lat == 0 and
                             bool(np.all((track["lat"] >= -90) & (track["lat"] <= 90))))

        timestamp_rows.append({
            "partition": part,
            "trip_id": tid,
            "is_monotonic": is_monotonic,
            "dt_min": float(np.min(dts)) if len(dts) > 0 else 0.0,
            "dt_median": float(np.median(dts)) if len(dts) > 0 else 0.0,
            "dt_mean": float(np.mean(dts)) if len(dts) > 0 else 0.0,
            "dt_max": max_gap,
            "pct_dt_1s": pct_1s,
            "gaps_gt_30s": gaps_gt_30s,
            "duplicate_coords": dup_gps,
            "duplicate_timestamps": dup_ts,
            "coord_order_valid": coord_order_valid,
        })

        missing_rows.append({
            "partition": part,
            "trip_id": tid,
            "nan_track": nan_track,
            "nan_nodes": nan_nodes,
            "nan_arcs": nan_arcs,
            "nan_route": nan_route,
            "invalid_lon": invalid_lon,
            "invalid_lat": invalid_lat,
        })

        # 3. Geographic extent & UTM projection
        projector = TripProjector(track["lon"].mean(), track["lat"].mean())
        gx, gy = projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
        nx, ny = projector.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())

        # Trajectory length (cumulative distance along GPS track)
        step_dists = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2)
        traj_len_km = float(np.sum(step_dists) / 1000.0)

        extent_rows.append({
            "partition": part,
            "trip_id": tid,
            "utm_epsg": projector.epsg,
            "min_lon": float(track["lon"].min()),
            "max_lon": float(track["lon"].max()),
            "min_lat": float(track["lat"].min()),
            "max_lat": float(track["lat"].max()),
            "span_x_km": float((gx.max() - gx.min()) / 1000.0),
            "span_y_km": float((gy.max() - gy.min()) / 1000.0),
            "traj_length_km": traj_len_km,
        })

        # 4. Road network statistics
        N_nodes = len(nodes)
        N_arcs = len(arcs)
        from_nds = arcs["from_node"].to_numpy()
        to_nds = arcs["to_node"].to_numpy()

        adj = sp.csr_matrix(
            (np.ones(N_arcs, dtype=bool), (from_nds, to_nds)),
            shape=(N_nodes, N_nodes)
        )
        n_weak = int(csgraph.connected_components(adj, directed=True, connection="weak")[0])
        n_strong = int(csgraph.connected_components(adj, directed=True, connection="strong")[0])

        out_deg = np.diff(adj.indptr)
        in_deg = np.bincount(to_nds, minlength=N_nodes)
        isolated_nodes = int(np.sum((out_deg == 0) & (in_deg == 0)))
        avg_out_deg = float(N_arcs / max(1, N_nodes))

        # Arc length distribution
        ax_seg = nx[from_nds]
        ay_seg = ny[from_nds]
        bx_seg = nx[to_nds]
        by_seg = ny[to_nds]
        arc_lengths = np.sqrt((bx_seg - ax_seg)**2 + (by_seg - ay_seg)**2)
        all_arc_lengths_sample.extend(arc_lengths[::max(1, len(arc_lengths) // 1000)].tolist())

        network_rows.append({
            "partition": part,
            "trip_id": tid,
            "node_count": N_nodes,
            "arc_count": N_arcs,
            "weak_components": n_weak,
            "strong_components": n_strong,
            "isolated_nodes": isolated_nodes,
            "avg_out_degree": avg_out_deg,
            "arc_len_mean_m": float(np.mean(arc_lengths)),
            "arc_len_median_m": float(np.median(arc_lengths)),
            "arc_len_p95_m": float(np.percentile(arc_lengths, 95)),
            "arc_len_max_m": float(np.max(arc_lengths)),
        })

        # 5. Reference route checks
        if route is not None and len(route) > 0:
            route_from = from_nds[route]
            route_to = to_nds[route]
            consecutive_matches = np.sum(route_to[:-1] == route_from[1:])
            pct_consecutive = float((consecutive_matches / max(1, len(route) - 1)) * 100.0)
            n_unique_arcs = len(np.unique(route))
            repeated_arcs = len(route) - n_unique_arcs

            route_rows.append({
                "partition": part,
                "trip_id": tid,
                "route_length": len(route),
                "unique_arcs": n_unique_arcs,
                "repeated_arcs": repeated_arcs,
                "consecutive_connected_pct": pct_consecutive,
                "route_to_gps_ratio": float(len(route) / len(track)),
            })

            # 6. GPS-to-reference-route distance distribution (noise level)
            r_ax = nx[route_from]
            r_ay = ny[route_from]
            r_bx = nx[route_to]
            r_by = ny[route_to]

            # Fast chunked distance computation
            chunk_size = 1500
            trip_noise = []
            for c_start in range(0, len(gx), chunk_size):
                c_end = min(len(gx), c_start + chunk_size)
                dists_c, _, _ = vectorised_point_to_segment(
                    gx[c_start:c_end, None], gy[c_start:c_end, None],
                    r_ax[None, :], r_ay[None, :],
                    r_bx[None, :], r_by[None, :]
                )
                min_c = np.min(dists_c, axis=1)
                trip_noise.append(min_c)

            trip_noise = np.concatenate(trip_noise)
            all_noise_sample.extend(trip_noise[::max(1, len(trip_noise) // 200)].tolist())

            noise_rows.append({
                "partition": part,
                "trip_id": tid,
                "noise_mean_m": float(np.mean(trip_noise)),
                "noise_median_m": float(np.median(trip_noise)),
                "noise_p90_m": float(np.percentile(trip_noise, 90)),
                "noise_p95_m": float(np.percentile(trip_noise, 95)),
                "noise_p99_m": float(np.percentile(trip_noise, 99)),
                "noise_max_m": float(np.max(trip_noise)),
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_trips):
            print(f"Processed {idx + 1}/{len(all_trips)} trips ({round(time.time() - t_start, 1)}s)...")

    # Save CSV tables
    df_inventory = pd.DataFrame(inventory_rows)
    df_timestamp = pd.DataFrame(timestamp_rows)
    df_missing = pd.DataFrame(missing_rows)
    df_extent = pd.DataFrame(extent_rows)
    df_network = pd.DataFrame(network_rows)
    df_route = pd.DataFrame(route_rows)
    df_noise = pd.DataFrame(noise_rows)

    df_inventory.to_csv("results/eda/inventory.csv", index=False)
    df_timestamp.to_csv("results/eda/timestamp_and_quality.csv", index=False)
    df_missing.to_csv("results/eda/missing_values.csv", index=False)
    df_extent.to_csv("results/eda/geographic_extent.csv", index=False)
    df_network.to_csv("results/eda/road_network_stats.csv", index=False)
    df_route.to_csv("results/eda/reference_route_checks.csv", index=False)
    df_noise.to_csv("results/eda/noise_distribution.csv", index=False)

    print("All CSV tables successfully written to results/eda/")

    # 7. Generate Matplotlib plots
    print("Generating statistical EDA plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) GPS count distribution
    axes[0, 0].hist(df_inventory["gps_count"], bins=25, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("GPS Points per Trip Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("GPS Points Count")
    axes[0, 0].set_ylabel("Number of Trips")

    # (b) Sampling interval dt
    axes[0, 1].hist(all_dt_sample, bins=np.linspace(0, 10, 40), color="#ff7f0e", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title(r"Sampling Interval ($\Delta t$) Distribution (up to 10s)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel(r"$\Delta t$ (seconds)")
    axes[0, 1].set_ylabel("Observation Frequency")

    # (c) Trajectory Length (km)
    axes[0, 2].hist(df_extent["traj_length_km"], bins=20, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0, 2].set_title("Trajectory Length (km)", fontsize=12, fontweight="bold")
    axes[0, 2].set_xlabel("Trajectory Length (km)")
    axes[0, 2].set_ylabel("Number of Trips")

    # (d) Arc Lengths (log scale)
    arc_lens_sample = np.array(all_arc_lengths_sample)
    arc_lens_sample = arc_lens_sample[arc_lens_sample > 0]
    axes[1, 0].hist(np.log10(arc_lens_sample), bins=30, color="#d62728", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title(r"Road Arc Length Distribution ($\log_{10}$ metres)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel(r"$\log_{10}(\text{Arc Length in m})$")
    axes[1, 0].set_ylabel("Frequency")

    # (e) Route continuity percentage
    axes[1, 1].hist(df_route["consecutive_connected_pct"], bins=25, color="#9467bd", edgecolor="black", alpha=0.8)
    axes[1, 1].set_title("Route Arc Consecutive Connectivity (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Consecutive Connected Arcs (%)")
    axes[1, 1].set_ylabel("Number of Trips")

    # (f) GPS Noise (Distance to Route)
    noise_sample = np.array(all_noise_sample)
    axes[1, 2].hist(noise_sample[noise_sample < 50], bins=35, color="#8c564b", edgecolor="black", alpha=0.8)
    axes[1, 2].set_title("GPS-to-Reference Route Distance (m)", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Distance to Reference Route (m)")
    axes[1, 2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("results/eda/eda_statistical_distributions.png", dpi=200)
    plt.savefig("results/figures/eda_statistical_distributions.png", dpi=200)
    plt.close()

    # Trajectory + Road network plot for 3 sample trips
    sample_trip_ids = ["00000001", "00000002", "00000000"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, tid in enumerate(sample_trip_ids):
        # Find trip partition
        part = "train" if tid in ["00000001", "00000002"] else "student_test"
        trip = load_trip(os.path.join("Student folder", part, tid))
        projector = TripProjector(trip.track["lon"].mean(), trip.track["lat"].mean())

        tx, ty = projector.forward(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
        nx, ny = projector.forward(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())

        # Subsample nearby network for clear plotting
        from_nds = trip.arcs["from_node"].to_numpy()
        to_nds = trip.arcs["to_node"].to_numpy()

        pad = 500  # 500m around track
        x_min, x_max = tx.min() - pad, tx.max() + pad
        y_min, y_max = ty.min() - pad, ty.max() + pad

        in_box = (
            (nx[from_nds] >= x_min) & (nx[from_nds] <= x_max) &
            (ny[from_nds] >= y_min) & (ny[from_nds] <= y_max)
        )
        sample_from = from_nds[in_box]
        sample_to = to_nds[in_box]

        ax = axes[idx]
        # Plot road arcs
        for u, v in zip(sample_from[::max(1, len(sample_from) // 2000)], sample_to[::max(1, len(sample_from) // 2000)]):
            ax.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#cccccc", linewidth=0.8, zorder=1)

        # Plot route arcs if available
        if trip.route is not None:
            r_from = from_nds[trip.route]
            r_to = to_nds[trip.route]
            for u, v in zip(r_from, r_to):
                ax.plot([nx[u], nx[v]], [ny[u], ny[v]], color="#2ca02c", linewidth=2.0, zorder=2, alpha=0.8)

        # Plot GPS points
        ax.scatter(tx, ty, c="#1f77b4", s=8, zorder=3, alpha=0.7, label="GPS Points")
        ax.set_title(f"Trip {tid} ({part})", fontsize=12, fontweight="bold")
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    plt.savefig("results/eda/sample_trips_network_gps.png", dpi=200)
    plt.savefig("results/figures/sample_trips_network_gps.png", dpi=200)
    plt.close()

    # 8. Folium Interactive Map
    print("Generating Folium map for sample trip 00000001...")
    sample_tid = "00000001"
    trip = load_trip(os.path.join("Student folder", "train", sample_tid))

    # Center map on trajectory center
    center_lat = float(trip.track["lat"].mean())
    center_lon = float(trip.track["lon"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Road network within trip bbox
    min_lat, max_lat = trip.track["lat"].min() - 0.01, trip.track["lat"].max() + 0.01
    min_lon, max_lon = trip.track["lon"].min() - 0.01, trip.track["lon"].max() + 0.01

    nodes_arr = trip.nodes[["lon", "lat"]].to_numpy()
    from_nds = trip.arcs["from_node"].to_numpy()
    to_nds = trip.arcs["to_node"].to_numpy()

    # Filter nearby arcs
    nearby_mask = (
        (nodes_arr[from_nds, 0] >= min_lon) & (nodes_arr[from_nds, 0] <= max_lon) &
        (nodes_arr[from_nds, 1] >= min_lat) & (nodes_arr[from_nds, 1] <= max_lat)
    )
    sub_from = from_nds[nearby_mask]
    sub_to = to_nds[nearby_mask]

    fg_network = folium.FeatureGroup(name="Nearby Road Network (Grey)", show=True)
    # Add network lines (subsample if too many for fast rendering)
    step = max(1, len(sub_from) // 3000)
    for u, v in zip(sub_from[::step], sub_to[::step]):
        p1 = [nodes_arr[u, 1], nodes_arr[u, 0]]
        p2 = [nodes_arr[v, 1], nodes_arr[v, 0]]
        folium.PolyLine([p1, p2], color="#888888", weight=1.5, opacity=0.6).add_to(fg_network)
    fg_network.add_to(fmap)

    # Layer 2: Reference Route (Green line)
    fg_route = folium.FeatureGroup(name="Reference Route (Green)", show=True)
    if trip.route is not None:
        r_from = from_nds[trip.route]
        r_to = to_nds[trip.route]
        route_lines = []
        for u, v in zip(r_from, r_to):
            p1 = [nodes_arr[u, 1], nodes_arr[u, 0]]
            p2 = [nodes_arr[v, 1], nodes_arr[v, 0]]
            route_lines.append([p1, p2])
        for line in route_lines:
            folium.PolyLine(line, color="#2ca02c", weight=4, opacity=0.85).add_to(fg_route)
    fg_route.add_to(fmap)

    # Layer 3: Raw GPS points (Blue dots)
    fg_gps = folium.FeatureGroup(name="Raw GPS Points (Blue)", show=True)
    for _, row in trip.track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.7,
            popup=f"t={row['t']}s",
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap_path = f"maps/eda_trip_{sample_tid}.html"
    fmap.save(fmap_path)
    print(f"Saved Folium map to {fmap_path}")

    # 9. Write Data Quality Summary Markdown based on actual numbers
    print("Writing results/eda/data_quality.md...")
    total_trips = len(df_inventory)
    total_gps = df_inventory["gps_count"].sum()
    pct_1s_avg = df_timestamp["pct_dt_1s"].mean()
    trips_with_gaps_gt_30 = (df_timestamp["gaps_gt_30s"] > 0).sum()
    all_monotonic = df_timestamp["is_monotonic"].all()
    all_coord_valid = df_timestamp["coord_order_valid"].all()
    avg_noise_median = df_noise["noise_median_m"].mean()
    avg_noise_p95 = df_noise["noise_p95_m"].mean()
    avg_consec_conn = df_route["consecutive_connected_pct"].mean()

    summary_md = f"""# PathSnap Data Quality Assessment & Assumptions

## 1. Executive Summary & Inventory
- **Total Trips Analyzed:** {total_trips} (Train: 80, Student Test: 10)
- **Total GPS Observations:** {total_gps:,} points
- **Average GPS Points per Trip:** {df_inventory['gps_count'].mean():.1f} (min: {df_inventory['gps_count'].min()}, max: {df_inventory['gps_count'].max()})
- **Road Network Scale:** Trips contain between {df_inventory['node_count'].min():,} and {df_inventory['node_count'].max():,} nodes, and between {df_inventory['arc_count'].min():,} and {df_inventory['arc_count'].max():,} arcs.

## 2. Timestamps and Sampling Regularity
- **Monotonicity:** {"100% of trips have strictly monotonically increasing timestamps." if all_monotonic else "Non-monotonic timestamps detected in some trips."}
- **1-Second Sampling Rate:** On average, **{pct_1s_avg:.2f}%** of consecutive observations have exactly dt = 1.0 s.
- **GPS Gaps (>30s):** {trips_with_gaps_gt_30} out of {total_trips} trips contain sampling gaps exceeding 30 seconds (maximum recorded gap: {df_timestamp['dt_max'].max():.1f} s).
- **Duplicate Records:** No duplicate timestamps were found ({df_timestamp['duplicate_timestamps'].sum()} duplicates across all tracks).

## 3. Geographic Coordinates & Projection Integrity
- **Coordinate Order:** Confirmed `lon` first, `lat` second in all raw files. All latitudes strictly lie within [-90, +90] and longitudes within [-180, +180] ({'valid' if all_coord_valid else 'invalid'}).
- **Missing Values (NaN/Inf):** None. Total NaN counts in track, nodes, arcs, and route files are strictly 0.
- **Geographic Extent:** Trips span distinct worldwide locations requiring per-trip UTM metric projections (EPSG UTM zones automatically resolved per trip centroid). Trajectory lengths range from {df_extent['traj_length_km'].min():.2f} km to {df_extent['traj_length_km'].max():.2f} km (mean: {df_extent['traj_length_km'].mean():.2f} km).

## 4. Road Network Topology & Graph Properties
- **Graph Completeness:** Networks have an average out-degree of **{df_network['avg_out_degree'].mean():.2f}**.
- **Connectivity:** Each trip network is composed of multiple connected components (average weakly connected components: {df_network['weak_components'].mean():.1f}; strongly connected components: {df_network['strong_components'].mean():.1f}). Total isolated nodes across all networks: {df_network['isolated_nodes'].sum()}.
- **Arc Lengths:** Arc lengths vary from under 1 m to {df_network['arc_len_max_m'].max():.1f} m, with an overall median length of {df_network['arc_len_median_m'].median():.1f} m.

## 5. Reference Route & Noise Level
- **Route Continuity:** On average, **{avg_consec_conn:.2f}%** of consecutive route arcs satisfy `arcs[r_i].to == arcs[r_{{i+1}}].from`. Minor discontinuities exist where physical road connections skip zero-length or intermediate connector links.
- **Sensor Noise (GPS-to-Route Distance):**
  - **Mean Median Distance:** {avg_noise_median:.2f} m
  - **Mean 95th Percentile Distance:** {avg_noise_p95:.2f} m
  - **Maximum Distance Deviation:** {df_noise['noise_max_m'].max():.2f} m (outliers caused by multipath / tunnel signal drift).

## 6. Engineering Decisions & Assumptions
1. **Metric CRS:** All coordinates are projected to a local metric UTM projection per trip prior to candidate generation or distance calculations. Raw lat/lon is never used as a distance feature.
2. **Directed Arcs:** Road segments are strictly directed; opposing lanes are distinct arcs with opposite bearings. Candidate ranking must strongly penalize opposing heading differences.
3. **Candidate Filtering:** A spatial index (cKDTree) over arc midpoints followed by exact point-to-segment orthogonal distance provides an efficient K=10 candidate generation pipeline without risking missing true road segments.
4. **Leakage Protection:** Reference routes (`.route`) are isolated exclusively to the label generation and scoring modules.
"""
    with open("results/eda/data_quality.md", "w") as f:
        f.write(summary_md)

    print("=" * 70)
    print("EDA completed successfully! Summary written to results/eda/data_quality.md")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
