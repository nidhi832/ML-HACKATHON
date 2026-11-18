"""
src/geo.py - Geographic projections, distance calculations, and bearing helpers.
"""

from typing import Tuple, Optional
import numpy as np
import pyproj


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Determine the UTM EPSG code given a longitude and latitude.
    EPSG 326xx for northern hemisphere, 327xx for southern hemisphere.
    """
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(zone, 60))
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


class TripProjector:
    """
    Projects geographic coordinates (lon, lat) to a local metric UTM coordinate system
    centered on the trip centroid, and vice versa.
    """
    def __init__(self, lon_center: float, lat_center: float):
        self.epsg = get_utm_epsg(lon_center, lat_center)
        self.transformer_fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self.transformer_inv = pyproj.Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project (lon, lat) in degrees to (x, y) in metres."""
        x, y = self.transformer_fwd.transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def inverse(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project (x, y) in metres back to (lon, lat) in degrees."""
        lon, lat = self.transformer_inv.transform(x, y)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def vectorised_point_to_segment(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised calculation of point-to-segment distance and projection fractions.
    
    Inputs can be 1D arrays of matching shape or broadcastable.
    Returns:
        dist: shortest distance in metres from point P to segment [A, B]
        proj_frac: clamped projection fraction in [0.0, 1.0] (0=at node A, 1=at node B)
        raw_frac: unclamped projection scalar t = ((P - A) . (B - A)) / ||B - A||^2
    """
    vx = bx - ax
    vy = by - ay
    l2 = vx * vx + vy * vy
    
    # Degenerate segments (where A == B)
    l2_safe = np.where(l2 == 0, 1.0, l2)
    
    wx = px - ax
    wy = py - ay
    raw_frac = (wx * vx + wy * vy) / l2_safe
    proj_frac = np.clip(raw_frac, 0.0, 1.0)
    proj_frac = np.where(l2 == 0, 0.0, proj_frac)
    
    # Nearest point on segment
    cx = ax + proj_frac * vx
    cy = ay + proj_frac * vy
    
    dx = px - cx
    dy = py - cy
    dist = np.sqrt(dx * dx + dy * dy)
    
    return dist, proj_frac, raw_frac


def calculate_bearing(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """
    Calculate grid bearing in degrees [0, 360) from (x1, y1) to (x2, y2).
    0 deg = North (+Y), 90 deg = East (+X), 180 deg = South (-Y), 270 deg = West (-X).
    """
    dx = x2 - x1
    dy = y2 - y1
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    return bearing


def calculate_gps_headings(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 1,
) -> np.ndarray:
    """
    Calculate GPS trajectory headings in degrees [0, 360) along the trajectory.
    Uses symmetric difference x[i+w] - x[i-w] for interior points,
    forward difference at start, backward difference at end.
    Propagates forward if stationary (distance < 0.1m).
    """
    n = len(x)
    headings = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return headings

    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)

    w = max(1, window)
    for i in range(n):
        idx_prev = max(0, i - w)
        idx_next = min(n - 1, i + w)
        if idx_next == idx_prev:
            idx_prev = max(0, i - 1)
            idx_next = min(n - 1, i + 1)
        dx[i] = x[idx_next] - x[idx_prev]
        dy[i] = y[idx_next] - y[idx_prev]

    dist_step = np.sqrt(dx * dx + dy * dy)
    headings = np.degrees(np.arctan2(dx, dy)) % 360.0

    # For points where vehicle was stationary, forward-fill previous heading
    last_h = 0.0
    for i in range(n):
        if dist_step[i] < 0.1:
            headings[i] = last_h
        else:
            last_h = headings[i]

    # Backward-fill initial stationary points if any
    for i in range(n - 2, -1, -1):
        if dist_step[i] < 0.1 and dist_step[i + 1] >= 0.1:
            headings[i] = headings[i + 1]

    return headings


def angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smallest signed angle difference (angle1 - angle2) in [-180, 180]
    and absolute difference |diff| in [0, 180], and cos(diff) in [-1, 1].
    
    Returns:
        abs_diff: |angle1 - angle2| in degrees [0, 180]
        cos_diff: cos(angle1 - angle2) in [-1, 1]
    """
    diff = (angle1 - angle2 + 180.0) % 360.0 - 180.0
    abs_diff = np.abs(diff)
    cos_diff = np.cos(np.radians(diff))
    return abs_diff, cos_diff
