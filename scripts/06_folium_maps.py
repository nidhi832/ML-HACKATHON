"""
PathSnap: Multi-Layer Interactive Folium Map Generator
Generates rich, layered interactive maps for Best, Typical, and Worst student-test trips.
Displays:
  - High-reliability base maps (Esri WorldStreetMap, CartoDB Light, CartoDB Voyager) - NO 403 BLOCKS!
  - Raw GPS observations (blue dots with timestamp/speed popups)
  - Nearby road network (grey thin lines)
  - Reference ground-truth route (green line)
  - Model predicted route (red dashed line)
  - Misclassified GPS observations (orange glowing circle markers)
Includes:
  - Legend box
  - LayerControl for toggling layers & base maps
  - Auto-fit bounding box
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium import plugins

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_io import load_trip, list_trips

STUDENT_TEST_DIR = PROJECT_ROOT / "Student folder" / "student_test"
RESULTS_DIR = PROJECT_ROOT / "results"
MAPS_DIR = PROJECT_ROOT / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)


def build_legend_html():
    """Build an HTML floating legend for Folium."""
    return """
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 230px; height: 175px; 
        background-color: white; border:2px solid #555; z-index:9999; 
        font-size:12px; font-family: Arial, sans-serif; padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 6px;
        opacity: 0.95;
    ">
        <b style="font-size:13px;">PathSnap Map Layers</b><br>
        <div style="margin-top: 6px; line-height: 1.6;">
            <i style="background: #2b5c8f; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px;"></i> Raw GPS Points<br>
            <i style="background: #27ae60; width: 16px; height: 4px; display: inline-block; margin-right: 6px; vertical-align: middle;"></i> Reference Route (Ground Truth)<br>
            <i style="background: #c0392b; width: 16px; height: 4px; display: inline-block; margin-right: 6px; vertical-align: middle; border-top: 2px dashed #c0392b;"></i> Predicted Route (Top-1)<br>
            <i style="background: #e67e22; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px;"></i> Misclassified Points<br>
            <i style="background: #7f8c8d; width: 16px; height: 2px; display: inline-block; margin-right: 6px; vertical-align: middle;"></i> Road Network (Arcs)<br>
        </div>
    </div>
    """


def make_map(trip_id: str, preds_df: pd.DataFrame, out_path: Path, prob_col: str = "prob_lightgbm") -> folium.Map:
    """Generate a multi-layer interactive map for a given trip with verified tile servers."""
    trip_dir = STUDENT_TEST_DIR / trip_id
    trip = load_trip(trip_dir)
    
    trip_preds = preds_df[preds_df["trip_id"] == trip_id].copy()
    if trip_preds.empty:
        raise ValueError(f"No predictions found for trip {trip_id}")
        
    # Top-1 candidate per observation
    idx_max = trip_preds.groupby("obs_idx")[prob_col].idxmax()
    top1 = trip_preds.loc[idx_max].sort_values("obs_idx").reset_index(drop=True)
    
    center_lat = float(trip.track["lat"].mean())
    center_lon = float(trip.track["lon"].mean())
    
    # Initialize map without default tiles to prevent OSM 403 policy block
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None
    )
    
    # Add reliable, high-performance base map tiles
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China",
        name="Esri WorldStreetMap (Clear Roads)",
        overlay=False,
        control=True
    ).add_to(fmap)
    
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CartoDB Light (High Contrast)",
        overlay=False,
        control=True
    ).add_to(fmap)
    
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CartoDB Voyager",
        overlay=False,
        control=True
    ).add_to(fmap)
    
    # Feature Groups
    fg_network = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=False)
    fg_ref = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
    fg_pred = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
    fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory", show=True)
    fg_misclass = folium.FeatureGroup(name="Misclassified GPS Points", show=True)
    
    nodes = np.asarray(trip.nodes[["lon", "lat"]].values if isinstance(trip.nodes, pd.DataFrame) else trip.nodes)
    arcs = np.asarray(trip.arcs[["from_node", "to_node"]].values if isinstance(trip.arcs, pd.DataFrame) else trip.arcs)
    
    lat_min, lat_max = trip.track["lat"].min() - 0.005, trip.track["lat"].max() + 0.005
    lon_min, lon_max = trip.track["lon"].min() - 0.005, trip.track["lon"].max() + 0.005
    
    network_lines = []
    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]
    
    u_lons = nodes[from_nodes, 0]
    u_lats = nodes[from_nodes, 1]
    v_lons = nodes[to_nodes, 0]
    v_lats = nodes[to_nodes, 1]
    
    in_bbox = (
        ((u_lats >= lat_min) & (u_lats <= lat_max) & (u_lons >= lon_min) & (u_lons <= lon_max)) |
        ((v_lats >= lat_min) & (v_lats <= lat_max) & (v_lons >= lon_min) & (v_lons <= lon_max))
    )
    valid_arc_indices = np.where(in_bbox)[0]
    
    if len(valid_arc_indices) > 10000:
        valid_arc_indices = np.random.choice(valid_arc_indices, size=10000, replace=False)
        
    for a_idx in valid_arc_indices:
        u = from_nodes[a_idx]
        v = to_nodes[a_idx]
        network_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        
    if network_lines:
        folium.PolyLine(
            network_lines,
            color="#7f8c8d",
            weight=1.5,
            opacity=0.45
        ).add_to(fg_network)
        
    # Reference Route
    if len(trip.route) > 0:
        ref_lines = []
        for r_arc in trip.route:
            if r_arc < len(arcs):
                u = arcs[r_arc, 0]
                v = arcs[r_arc, 1]
                ref_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        if ref_lines:
            folium.PolyLine(
                ref_lines,
                color="#27ae60",
                weight=5,
                opacity=0.85,
                tooltip=f"Reference Route ({len(trip.route)} arcs)"
            ).add_to(fg_ref)
            
    # Predicted Route
    pred_lines = []
    pred_arcs = top1["arc_id"].values
    for p_arc in pred_arcs:
        if p_arc < len(arcs):
            u = arcs[p_arc, 0]
            v = arcs[p_arc, 1]
            pred_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
    if pred_lines:
        folium.PolyLine(
            pred_lines,
            color="#c0392b",
            weight=3.5,
            opacity=0.9,
            dash_array="6, 4",
            tooltip="PathSnap Top-1 Predicted Route"
        ).add_to(fg_pred)
        
    # Raw GPS Observations & Misclassifications
    for _, row in top1.iterrows():
        obs_idx = int(row["obs_idx"])
        lat = float(trip.track.iloc[obs_idx]["lat"])
        lon = float(trip.track.iloc[obs_idx]["lon"])
        t = float(trip.track.iloc[obs_idx]["t"]) if "t" in trip.track.columns else float(trip.track.iloc[obs_idx].get("time", 0.0))
        is_correct = bool(row["y"] == 1)
        prob = float(row[prob_col])
        
        status_span = '<span style="color:green">CORRECT</span>' if is_correct else '<span style="color:red">MISMATCH</span>'
        popup_text = f"""
        <b>Obs Index:</b> {obs_idx}<br>
        <b>Time:</b> {t:.1f}s<br>
        <b>Pred Arc:</b> {int(row['arc_id'])} (P={prob:.3f})<br>
        <b>Match:</b> {status_span}
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#2b5c8f",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(fg_gps)
        
        if not is_correct:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#e67e22",
                fill=True,
                fill_color="#f39c12",
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_text + "<br><b>Flag:</b> Model Mismatch", max_width=250)
            ).add_to(fg_misclass)

    fg_network.add_to(fmap)
    fg_ref.add_to(fmap)
    fg_pred.add_to(fmap)
    fg_gps.add_to(fmap)
    fg_misclass.add_to(fmap)
    
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(build_legend_html()))
    
    sw = [float(trip.track["lat"].min()), float(trip.track["lon"].min())]
    ne = [float(trip.track["lat"].max()), float(trip.track["lon"].max())]
    fmap.fit_bounds([sw, ne], padding=[20, 20])
    
    fmap.save(str(out_path))
    print(f"Generated clean interactive map -> {out_path}")
    return fmap


def main():
    print("=" * 80)
    print("PathSnap: Generating Folium Interactive Maps (Esri / CartoDB Base Tiles)")
    print("=" * 80)
    
    preds_file = RESULTS_DIR / "student_test_predictions.parquet"
    if not preds_file.exists():
        print(f"ERROR: {preds_file} does not exist yet.")
        return
        
    preds_df = pd.read_parquet(preds_file)
    print(f"Loaded student_test predictions ({len(preds_df):,} rows)")
    
    prob_col = "prob_lightgbm" if "prob_lightgbm" in preds_df.columns else "prob_random_forest"
    
    idx_max = preds_df.groupby(["trip_id", "obs_idx"])[prob_col].idxmax()
    top1 = preds_df.loc[idx_max].copy()
    trip_accs = top1.groupby("trip_id")["y"].mean().sort_values(ascending=False)
    
    best_trip_id = trip_accs.index[0]
    worst_trip_id = trip_accs.index[-1]
    median_idx = len(trip_accs) // 2
    typical_trip_id = trip_accs.index[median_idx]
    
    print(f"Generating maps for Best ({best_trip_id}), Typical ({typical_trip_id}), and Worst ({worst_trip_id})...")
    
    make_map(best_trip_id, preds_df, MAPS_DIR / f"student_test_best_{best_trip_id}.html", prob_col)
    make_map(typical_trip_id, preds_df, MAPS_DIR / f"student_test_typical_{typical_trip_id}.html", prob_col)
    make_map(worst_trip_id, preds_df, MAPS_DIR / f"student_test_worst_{worst_trip_id}.html", prob_col)
    
    print("\nAll maps regenerated with clean Esri WorldStreetMap & CartoDB tiles (zero 403 blocks)!")


if __name__ == "__main__":
    main()
