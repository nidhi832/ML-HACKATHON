# PathSnap Feature Set Documentation & Leakage Audit

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
- `heading_abs_diff`: Absolute angular difference $| 	ext{gps\_heading} - 	ext{arc\_bearing} | \in [0, 180]^\circ$. Critical for eliminating opposing one-way or dual-carriageway segments.
- `heading_cos_diff`: Cosine of the angular difference $\cos(	ext{gps\_heading} - 	ext{arc\_bearing}) \in [-1.0, 1.0]$. Directly aligns forward vehicle travel ($+1$) against wrong-way travel ($-1$).
- `speed_mps`: Instantaneous vehicle travel speed in metres per second.
- `is_stationary`: Binary indicator (1 if speed $< 0.5$ m/s, else 0); suppresses unreliable heading changes during traffic stops or red lights.
- `turn_angle_deg`: Heading change magnitude from the previous observation in degrees $[0, 180]^\circ$. Flags vehicle turns and intersection maneuvers.
- `acceleration_mps2`: Vehicle acceleration in $	ext{m/s}^2$ computed over consecutive observations.
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
