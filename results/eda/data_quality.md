# PathSnap Data Quality Assessment & Assumptions

## 1. Executive Summary & Inventory
- **Total Trips Analyzed:** 90 (Train: 80, Student Test: 10)
- **Total GPS Observations:** 227,577 points
- **Average GPS Points per Trip:** 2528.6 (min: 262, max: 19313)
- **Road Network Scale:** Trips contain between 299 and 155,003 nodes, and between 649 and 307,392 arcs.

## 2. Timestamps and Sampling Regularity
- **Monotonicity:** 100% of trips have strictly monotonically increasing timestamps.
- **1-Second Sampling Rate:** On average, **94.32%** of consecutive observations have exactly dt = 1.0 s.
- **GPS Gaps (>30s):** 30 out of 90 trips contain sampling gaps exceeding 30 seconds (maximum recorded gap: 153064.0 s).
- **Duplicate Records:** No duplicate timestamps were found (0 duplicates across all tracks).

## 3. Geographic Coordinates & Projection Integrity
- **Coordinate Order:** Confirmed `lon` first, `lat` second in all raw files. All latitudes strictly lie within [-90, +90] and longitudes within [-180, +180] (valid).
- **Missing Values (NaN/Inf):** None. Total NaN counts in track, nodes, arcs, and route files are strictly 0.
- **Geographic Extent:** Trips span distinct worldwide locations requiring per-trip UTM metric projections (EPSG UTM zones automatically resolved per trip centroid). Trajectory lengths range from 5.56 km to 169.55 km (mean: 36.43 km).

## 4. Road Network Topology & Graph Properties
- **Graph Completeness:** Networks have an average out-degree of **2.02**.
- **Connectivity:** Each trip network is composed of multiple connected components (average weakly connected components: 37.2; strongly connected components: 233.5). Total isolated nodes across all networks: 0.
- **Arc Lengths:** Arc lengths vary from under 1 m to 9357.6 m, with an overall median length of 25.1 m.

## 5. Reference Route & Noise Level
- **Route Continuity:** On average, **99.94%** of consecutive route arcs satisfy `arcs[r_i].to == arcs[r_{i+1}].from`. Minor discontinuities exist where physical road connections skip zero-length or intermediate connector links.
- **Sensor Noise (GPS-to-Route Distance):**
  - **Mean Median Distance:** 7.74 m
  - **Mean 95th Percentile Distance:** 22.91 m
  - **Maximum Distance Deviation:** 569.66 m (outliers caused by multipath / tunnel signal drift).

## 6. Engineering Decisions & Assumptions
1. **Metric CRS:** All coordinates are projected to a local metric UTM projection per trip prior to candidate generation or distance calculations. Raw lat/lon is never used as a distance feature.
2. **Directed Arcs:** Road segments are strictly directed; opposing lanes are distinct arcs with opposite bearings. Candidate ranking must strongly penalize opposing heading differences.
3. **Candidate Filtering:** A spatial index (cKDTree) over arc midpoints followed by exact point-to-segment orthogonal distance provides an efficient K=10 candidate generation pipeline without risking missing true road segments.
4. **Leakage Protection:** Reference routes (`.route`) are isolated exclusively to the label generation and scoring modules.
