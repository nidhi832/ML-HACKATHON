"""
src/data_io.py - Data loading and persistence utilities for PathSnap.
"""

import os
from typing import NamedTuple, Optional, Tuple, List
import numpy as np
import pandas as pd


class TripData(NamedTuple):
    track: pd.DataFrame       # columns: ['lon', 'lat', 't']
    nodes: pd.DataFrame       # columns: ['lon', 'lat'], index is node_id
    arcs: pd.DataFrame        # columns: ['from_node', 'to_node'], index is arc_id
    route: Optional[np.ndarray] # array of arc_ids (int32)
    trip_id: str


def load_trip(trip_dir: str) -> TripData:
    """
    Load a single trip directory containing <id>.track, <id>.nodes, <id>.arcs, and optionally <id>.route.
    
    Returns a TripData namedtuple with attributes: track, nodes, arcs, route, trip_id.
    Can also be unpacked as (track, nodes, arcs, route).
    """
    trip_dir = os.path.abspath(trip_dir)
    trip_id = os.path.basename(trip_dir)

    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Track file not found: {track_file}")
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not os.path.exists(arcs_file):
        raise FileNotFoundError(f"Arcs file not found: {arcs_file}")

    # Load track: lon, lat, timestamp_seconds
    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
        engine="c",
    )

    # Load nodes: lon, lat
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
        engine="c",
    )

    # Load arcs: from_node, to_node
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int32, "to_node": np.int32},
        engine="c",
    )

    # Load route if it exists
    route = None
    if os.path.exists(route_file):
        route_df = pd.read_csv(
            route_file,
            header=None,
            names=["arc_id"],
            dtype={"arc_id": np.int32},
            engine="c",
        )
        route = route_df["arc_id"].to_numpy(dtype=np.int32)

    return TripData(track=track, nodes=nodes, arcs=arcs, route=route, trip_id=trip_id)


def list_trips(data_root: str = "Student folder", partition: str = "train") -> List[Tuple[str, str]]:
    """
    List all trips in a given partition ('train' or 'student_test').
    Returns list of (trip_id, trip_dir_path).
    """
    partition_dir = os.path.join(data_root, partition)
    if not os.path.exists(partition_dir):
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")

    trip_ids = sorted([
        d for d in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, d))
    ])
    return [(tid, os.path.join(partition_dir, tid)) for tid in trip_ids]


def save_parquet(df: pd.DataFrame, path: str):
    """Save dataframe to parquet format with zstd compression."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def load_parquet(path: str) -> pd.DataFrame:
    """Load dataframe from parquet format."""
    return pd.read_parquet(path, engine="pyarrow")
