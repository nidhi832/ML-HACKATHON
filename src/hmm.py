"""
PathSnap: Hidden Markov Model (HMM) & Viterbi Sequence Decoder (Person A)
Decodes per-observation candidate road arcs into continuous, physically valid trajectories.

Mathematical Formulation:
- Hidden State at time t:
    s_t in {c_{t, 1}, ..., c_{t, K}} — candidate road arc ID preserving exact
    candidate-to-road-segment correspondence.
- Emission Probability:
    Derived from calibrated ML candidate probabilities P(arc | GPS_t, features),
    floored at `emission_floor` for numerical safety, normalized across the K
    candidates at step t, and scored in log space:
        log P(z_t | s_t = c_{t, k}) = log(p_tilde_{t, k} / sum_j p_tilde_{t, j}) / T_em
- Transition Probability:
    Enforces directed road network topology between consecutive candidate arcs
    a_{t-1} = (u_prev -> v_prev) and a_t = (u_curr -> v_curr):
      1. Same arc (a_{t-1} == a_t):                           same_arc_log_prob
      2. Forward direct edge (v_prev == u_curr, u_prev != v_curr): direct_connected_log_prob
      3. Instantaneous 180-deg reverse-arc U-turn
         (v_prev == u_curr and u_prev == v_curr):             reverse_arc_log_prob
      4. 2-hop directed path (v_prev -> w -> u_curr):         two_hop_log_prob
      5. Optional 3-hop directed path:                        three_hop_log_prob
      6. Disconnected / invalid transition:                   disconnected_penalty (finite)
- GPS Gap & Dropout Handling:
    When `timestamps` (or `dt_prev`) shows a sensor dropout (dt > gap_threshold_sec)
    or `track_utm` shows a spatial jump (delta_d > spatial_gap_m), the effective
    transition weight is smoothly softened:
        w_eff = transition_weight / (1.0 + gap_ratio)
    allowing the decoder to bridge dropouts via emission evidence without stalling.
- Missing Candidate Handling:
    If an observation index has zero/NaN probabilities, emissions safely fall back
    to uniform 1/K. If `num_observations` is provided and an observation index has
    no candidate rows at all, its arc is imputed from the nearest decoded neighbor
    so every GPS observation receives a valid road segment.
- Decoding (Viterbi Dynamic Programming):
    Log-space trellis with per-step max-normalization (preventing underflow on
    4,000+ observation trips while preserving exact argmax and backpointers).
"""

from bisect import bisect_left
from collections import Counter, defaultdict
import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


class ViterbiMapMatcher:
    """
    Hidden Markov Model (HMM) map matcher using log-space Viterbi decoding
    over directed road-network candidate graphs.
    """

    def __init__(
        self,
        same_arc_log_prob: float = -0.1,
        direct_connected_log_prob: float = -0.5,
        reverse_arc_log_prob: float = -10.0,
        two_hop_log_prob: float = -2.5,
        three_hop_log_prob: Optional[float] = None,
        disconnected_penalty: float = -25.0,
        emission_weight: float = 1.0,
        transition_weight: float = 2.0,
        emission_floor: float = 1e-6,
        emission_temperature: float = 1.0,
        gap_threshold_sec: float = 30.0,
        spatial_gap_m: float = 250.0,
    ):
        self.same_arc_log_prob = float(same_arc_log_prob)
        self.direct_connected_log_prob = float(direct_connected_log_prob)
        self.reverse_arc_log_prob = float(reverse_arc_log_prob)
        self.two_hop_log_prob = float(two_hop_log_prob)
        self.three_hop_log_prob = None if three_hop_log_prob is None else float(three_hop_log_prob)
        self.disconnected_penalty = float(disconnected_penalty)
        self.emission_weight = float(emission_weight)
        self.transition_weight = float(transition_weight)
        self.emission_floor = float(emission_floor)
        self.emission_temperature = max(1e-6, float(emission_temperature))
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.spatial_gap_m = float(spatial_gap_m)

    def get_config(self) -> Dict[str, Any]:
        """Return serializable decoder hyperparameter configuration."""
        return {
            "same_arc_log_prob": self.same_arc_log_prob,
            "direct_connected_log_prob": self.direct_connected_log_prob,
            "reverse_arc_log_prob": self.reverse_arc_log_prob,
            "two_hop_log_prob": self.two_hop_log_prob,
            "three_hop_log_prob": self.three_hop_log_prob,
            "disconnected_penalty": self.disconnected_penalty,
            "emission_weight": self.emission_weight,
            "transition_weight": self.transition_weight,
            "emission_floor": self.emission_floor,
            "emission_temperature": self.emission_temperature,
            "gap_threshold_sec": self.gap_threshold_sec,
            "spatial_gap_m": self.spatial_gap_m,
        }

    def build_network_graph(self, arcs: np.ndarray) -> Dict[int, List[int]]:
        """
        Build directed outgoing-arc adjacency lookup:
            out_arcs_by_node[from_node] = list of arc_ids originating at from_node.
        """
        out_arcs_by_node: Dict[int, List[int]] = defaultdict(list)
        from_nodes = arcs[:, 0]
        for arc_id in range(len(arcs)):
            out_arcs_by_node[int(from_nodes[arc_id])].append(arc_id)
        return out_arcs_by_node

    @staticmethod
    def build_node_adjacency(arcs: np.ndarray) -> Dict[int, Set[int]]:
        """
        Build directed node-to-node adjacency sets for O(1) 2-hop reachability checks:
            out_nodes[u] = set of nodes v such that directed arc (u -> v) exists.
        """
        out_nodes: Dict[int, Set[int]] = defaultdict(set)
        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]
        for i in range(len(arcs)):
            out_nodes[int(from_nodes[i])].add(int(to_nodes[i]))
        return dict(out_nodes)

    def _compute_log_emissions(self, raw_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw candidate probabilities into normalized probabilities and
        log-emission scores, handling NaNs, infinities, and all-zero rows safely.
        """
        probs = np.asarray(raw_probs, dtype=np.float64)
        if probs.size == 0:
            return probs, probs

        # Replace non-finite values with floor
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, self.emission_floor, 1.0)
        p_sum = float(probs.sum())
        if p_sum <= 0.0 or not np.isfinite(p_sum):
            norm_probs = np.full(probs.shape, 1.0 / len(probs), dtype=np.float64)
        else:
            norm_probs = probs / p_sum

        log_emissions = np.log(norm_probs) / self.emission_temperature
        return norm_probs, log_emissions

    def _effective_transition_weight(
        self,
        obs_prev: int,
        obs_curr: int,
        timestamps: Optional[np.ndarray],
        track_utm: Optional[np.ndarray],
        dt_override: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """
        Compute effective transition weight for step (obs_prev -> obs_curr).
        Smoothly softens transition constraints across large temporal GPS gaps
        or large spatial jumps so the Viterbi decoder can re-lock using emissions.
        """
        gap_ratio = 0.0

        # Temporal gap check
        dt = dt_override
        if dt is None and timestamps is not None:
            if obs_prev < len(timestamps) and obs_curr < len(timestamps):
                dt = float(timestamps[obs_curr] - timestamps[obs_prev])

        if dt is not None and np.isfinite(dt) and dt > self.gap_threshold_sec and self.gap_threshold_sec > 0:
            gap_ratio = max(gap_ratio, dt / self.gap_threshold_sec)

        # Spatial jump check (when metric UTM coordinates are supplied)
        if track_utm is not None and self.spatial_gap_m > 0:
            if obs_prev < len(track_utm) and obs_curr < len(track_utm):
                dx = float(track_utm[obs_curr, 0] - track_utm[obs_prev, 0])
                dy = float(track_utm[obs_curr, 1] - track_utm[obs_prev, 1])
                step_dist = float(np.hypot(dx, dy))
                if step_dist > self.spatial_gap_m:
                    gap_ratio = max(gap_ratio, step_dist / self.spatial_gap_m)

        if gap_ratio > 0.0:
            return self.transition_weight / (1.0 + gap_ratio), True

        return self.transition_weight, False

    def decode_trip(
        self,
        obs_candidates: pd.DataFrame,
        arcs: np.ndarray,
        track_utm: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        num_observations: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Apply log-space Viterbi decoding over the candidate sequence for a single trip.

        Args:
            obs_candidates: DataFrame with required columns ['obs_idx', 'arc_id', 'prob'].
                            Optional column 'dt_prev' is used for gap detection if
                            `timestamps` is not explicitly provided.
            arcs: Directed road arcs array of shape (N_arcs, 2) with [from_node, to_node].
            track_utm: Optional (N_obs, 2) metric UTM coordinates for spatial gap checks.
            timestamps: Optional (N_obs,) timestamp array in seconds for GPS gap handling.
            num_observations: Optional total GPS observation count N. If provided and any
                              obs_idx in 0..N-1 has missing candidates, those indices are
                              imputed cleanly so every GPS point gets a predicted arc.
            return_diagnostics: If True, return (decoded_df, diagnostics_dict).

        Returns:
            DataFrame with columns ['obs_idx', 'viterbi_arc_id', 'viterbi_prob']
            (or tuple (df, diagnostics) when return_diagnostics=True).
        """
        empty_df = pd.DataFrame(columns=["obs_idx", "viterbi_arc_id", "viterbi_prob"])
        diagnostics: Dict[str, Any] = {
            "num_observations_decoded": 0,
            "missing_candidate_obs": 0,
            "gap_softened_steps": 0,
            "transition_counts": {
                "same_arc": 0,
                "direct_forward": 0,
                "reverse_uturn": 0,
                "two_hop": 0,
                "three_hop": 0,
                "disconnected": 0,
            },
        }

        if obs_candidates is None or len(obs_candidates) == 0:
            return (empty_df, diagnostics) if return_diagnostics else empty_df

        arcs_arr = np.asarray(arcs, dtype=np.int64)
        num_arcs = len(arcs_arr)
        from_nodes_all = arcs_arr[:, 0]
        to_nodes_all = arcs_arr[:, 1]

        # Build fast O(1) node adjacency for 2-hop (and lazy 3-hop) checks
        out_nodes = self.build_node_adjacency(arcs_arr)
        hop2_node_cache: Dict[int, Set[int]] = {}

        def get_two_hop_nodes(node_v: int) -> Set[int]:
            cached = hop2_node_cache.get(node_v)
            if cached is None:
                reachable: Set[int] = set()
                for mid in out_nodes.get(node_v, ()):
                    reachable.update(out_nodes.get(mid, ()))
                hop2_node_cache[node_v] = reachable
                return reachable
            return cached

        # Sort & partition candidates by observation index without slow pandas per-group slicing
        has_dt_col = "dt_prev" in obs_candidates.columns
        obs_col = obs_candidates["obs_idx"].to_numpy(dtype=np.int64)
        arc_col = obs_candidates["arc_id"].to_numpy(dtype=np.int64)
        prob_col = obs_candidates["prob"].to_numpy(dtype=np.float64)
        dt_col = obs_candidates["dt_prev"].to_numpy(dtype=np.float64) if has_dt_col else None

        # Ensure ascending order by obs_idx (preserve stable candidate order within each obs)
        if len(obs_col) > 1 and np.any(obs_col[1:] < obs_col[:-1]):
            order = np.argsort(obs_col, kind="mergesort")
            obs_col = obs_col[order]
            arc_col = arc_col[order]
            prob_col = prob_col[order]
            if dt_col is not None:
                dt_col = dt_col[order]

        unique_obs, split_starts = np.unique(obs_col, return_index=True)
        T = len(unique_obs)
        if T == 0:
            return (empty_df, diagnostics) if return_diagnostics else empty_df

        split_ends = np.append(split_starts[1:], len(obs_col))

        arc_arrays: List[np.ndarray] = []
        norm_prob_arrays: List[np.ndarray] = []
        log_em_arrays: List[np.ndarray] = []
        dt_values: List[Optional[float]] = []

        for t in range(T):
            s_idx, e_idx = int(split_starts[t]), int(split_ends[t])
            c_arcs = arc_col[s_idx:e_idx]
            c_probs = prob_col[s_idx:e_idx]

            # Filter out-of-range arc IDs defensively
            valid_mask = (c_arcs >= 0) & (c_arcs < num_arcs)
            if not np.all(valid_mask):
                c_arcs = c_arcs[valid_mask]
                c_probs = c_probs[valid_mask]
            if len(c_arcs) == 0:
                c_arcs = np.array([0], dtype=np.int64)
                c_probs = np.array([1.0], dtype=np.float64)

            n_probs, l_em = self._compute_log_emissions(c_probs)
            arc_arrays.append(c_arcs)
            norm_prob_arrays.append(n_probs)
            log_em_arrays.append(l_em * self.emission_weight)
            dt_values.append(float(dt_col[s_idx]) if dt_col is not None else None)

        # Initialize Viterbi trellis at t = 0 with max-normalization
        vit_prev = log_em_arrays[0].copy()
        vit_prev -= float(np.max(vit_prev))
        backpointers: List[np.ndarray] = []

        use_three_hop = self.three_hop_log_prob is not None
        gap_softened_steps = 0

        for t in range(1, T):
            prev_arcs = arc_arrays[t - 1]
            curr_arcs = arc_arrays[t]
            log_em_curr = log_em_arrays[t]

            Kp = len(prev_arcs)
            Kc = len(curr_arcs)

            obs_prev = int(unique_obs[t - 1])
            obs_curr = int(unique_obs[t])
            eff_tr_w, was_gap = self._effective_transition_weight(
                obs_prev=obs_prev,
                obs_curr=obs_curr,
                timestamps=timestamps,
                track_utm=track_utm,
                dt_override=dt_values[t] if timestamps is None else None,
            )
            if was_gap:
                gap_softened_steps += 1

            # Compute (Kp, Kc) transition log-probability matrix
            tmat = np.full((Kp, Kc), self.disconnected_penalty, dtype=np.float64)

            u_prev_arr = from_nodes_all[prev_arcs]
            v_prev_arr = to_nodes_all[prev_arcs]
            u_curr_arr = from_nodes_all[curr_arcs]
            v_curr_arr = to_nodes_all[curr_arcs]

            for i in range(Kp):
                a_prev = int(prev_arcs[i])
                u_prev = int(u_prev_arr[i])
                v_prev = int(v_prev_arr[i])
                out_vp = out_nodes.get(v_prev)

                for j in range(Kc):
                    a_curr = int(curr_arcs[j])
                    if a_prev == a_curr:
                        tmat[i, j] = self.same_arc_log_prob
                    else:
                        u_curr = int(u_curr_arr[j])
                        v_curr = int(v_curr_arr[j])
                        if v_prev == u_curr:
                            if u_prev == v_curr:
                                # Instantaneous 180-deg reverse-arc U-turn (u -> v -> u)
                                tmat[i, j] = self.reverse_arc_log_prob
                            else:
                                # Direct forward edge (v_prev == u_curr)
                                tmat[i, j] = self.direct_connected_log_prob
                        elif out_vp is not None and u_curr in out_vp:
                            # 2-hop directed path: v_prev -> w -> u_curr
                            tmat[i, j] = self.two_hop_log_prob
                        elif use_three_hop and u_curr in get_two_hop_nodes(v_prev):
                            tmat[i, j] = float(self.three_hop_log_prob)

            # Vectorized Viterbi recurrence:
            # V[t, j] = log_em[t, j] + max_i (V[t-1, i] + eff_tr_w * T[i, j])
            scores = vit_prev[:, None] + eff_tr_w * tmat
            bp = np.argmax(scores, axis=0).astype(np.int32)
            vit_curr = scores[bp, np.arange(Kc)] + log_em_curr

            # Per-step max-normalization prevents floating-point underflow on long trips
            vit_curr -= float(np.max(vit_curr))

            vit_prev = vit_curr
            backpointers.append(bp)

        # Backtrack globally optimal hidden-state path
        best_idx = int(np.argmax(vit_prev))
        best_state_indices = [best_idx]
        for t in range(T - 1, 0, -1):
            best_idx = int(backpointers[t - 1][best_idx])
            best_state_indices.append(best_idx)
        best_state_indices.reverse()

        decoded_obs = [int(o) for o in unique_obs]
        decoded_arcs = [int(arc_arrays[t][best_state_indices[t]]) for t in range(T)]
        decoded_probs = [float(norm_prob_arrays[t][best_state_indices[t]]) for t in range(T)]

        # Handle missing observations (if any obs_idx in 0..N-1 had 0 candidates)
        missing_count = 0
        expected_n = num_observations
        if expected_n is None and timestamps is not None and len(timestamps) > T:
            expected_n = len(timestamps)

        if expected_n is not None and expected_n > T:
            obs_to_decoded = {
                decoded_obs[i]: (decoded_arcs[i], decoded_probs[i])
                for i in range(T)
            }
            full_obs: List[int] = []
            full_arcs: List[int] = []
            full_probs: List[float] = []
            last_arc = decoded_arcs[0]

            for o_idx in range(expected_n):
                if o_idx in obs_to_decoded:
                    a_id, pr = obs_to_decoded[o_idx]
                    last_arc = a_id
                    full_obs.append(o_idx)
                    full_arcs.append(a_id)
                    full_probs.append(pr)
                else:
                    missing_count += 1
                    full_obs.append(o_idx)
                    full_arcs.append(last_arc)
                    full_probs.append(0.0)

            decoded_obs = full_obs
            decoded_arcs = full_arcs
            decoded_probs = full_probs

        result_df = pd.DataFrame({
            "obs_idx": decoded_obs,
            "viterbi_arc_id": decoded_arcs,
            "viterbi_prob": decoded_probs,
        })

        if not return_diagnostics:
            return result_df

        # Summarize transition breakdown for diagnostics
        t_counts = diagnostics["transition_counts"]
        for t in range(1, len(decoded_arcs)):
            a_p = decoded_arcs[t - 1]
            a_c = decoded_arcs[t]
            if a_p == a_c:
                t_counts["same_arc"] += 1
            else:
                u_p, v_p = int(from_nodes_all[a_p]), int(to_nodes_all[a_p])
                u_c, v_c = int(from_nodes_all[a_c]), int(to_nodes_all[a_c])
                if v_p == u_c:
                    if u_p == v_c:
                        t_counts["reverse_uturn"] += 1
                    else:
                        t_counts["direct_forward"] += 1
                elif u_c in out_nodes.get(v_p, ()):
                    t_counts["two_hop"] += 1
                elif use_three_hop and u_c in get_two_hop_nodes(v_p):
                    t_counts["three_hop"] += 1
                else:
                    t_counts["disconnected"] += 1

        diagnostics["num_observations_decoded"] = len(decoded_arcs)
        diagnostics["missing_candidate_obs"] = missing_count
        diagnostics["gap_softened_steps"] = gap_softened_steps
        return result_df, diagnostics


# ---------------------------------------------------------------------------
# Sequence-Level Route Reconstruction Metrics (used by evaluation scripts)
# ---------------------------------------------------------------------------

def compress_route(arc_sequence: Sequence[int]) -> List[int]:
    """Compress consecutive duplicate arc IDs into the traversed road arc sequence."""
    return [int(k) for k, _ in itertools.groupby(arc_sequence)]


def compute_route_continuity_stats(
    arc_sequence: Sequence[int],
    arcs: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze physical road-network continuity of an observation-level arc sequence.
    Classifies every consecutive transition into:
      - same_arc:       a_{t-1} == a_t
      - direct_forward: to(a_{t-1}) == from(a_t) and not reverse U-turn
      - reverse_uturn:  to(a_{t-1}) == from(a_t) and from(a_{t-1}) == to(a_t)
      - two_hop:        2-hop directed path exists in network
      - disconnected:   no <= 2-hop directed path
    """
    seq = np.asarray(arc_sequence, dtype=np.int64)
    n = len(seq)
    if n <= 1:
        return {
            "total_transitions": 0,
            "valid_transitions": 0,
            "route_continuity_pct": 100.0,
            "forward_continuity_pct": 100.0,
            "two_hop_continuity_pct": 100.0,
            "same_arc_count": 0,
            "direct_forward_count": 0,
            "reverse_uturn_count": 0,
            "two_hop_count": 0,
            "disconnected_count": 0,
        }

    arcs_arr = np.asarray(arcs, dtype=np.int64)
    from_nodes = arcs_arr[:, 0]
    to_nodes = arcs_arr[:, 1]
    out_nodes = ViterbiMapMatcher.build_node_adjacency(arcs_arr)

    same_cnt = 0
    fwd_cnt = 0
    uturn_cnt = 0
    two_hop_cnt = 0
    disc_cnt = 0

    for t in range(1, n):
        a_p = int(seq[t - 1])
        a_c = int(seq[t])
        if a_p == a_c:
            same_cnt += 1
        else:
            u_p, v_p = int(from_nodes[a_p]), int(to_nodes[a_p])
            u_c, v_c = int(from_nodes[a_c]), int(to_nodes[a_c])
            if v_p == u_c:
                if u_p == v_c:
                    uturn_cnt += 1
                else:
                    fwd_cnt += 1
            elif u_c in out_nodes.get(v_p, ()):
                two_hop_cnt += 1
            else:
                disc_cnt += 1

    total = n - 1
    valid_graph = same_cnt + fwd_cnt + uturn_cnt
    valid_forward = same_cnt + fwd_cnt
    valid_2hop = valid_graph + two_hop_cnt

    return {
        "total_transitions": total,
        "valid_transitions": valid_graph,
        "route_continuity_pct": (valid_graph / total) * 100.0,
        "forward_continuity_pct": (valid_forward / total) * 100.0,
        "two_hop_continuity_pct": (valid_2hop / total) * 100.0,
        "same_arc_count": same_cnt,
        "direct_forward_count": fwd_cnt,
        "reverse_uturn_count": uturn_cnt,
        "two_hop_count": two_hop_cnt,
        "disconnected_count": disc_cnt,
    }


def _exact_lcs_length(seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
    """
    Exact Longest Common Subsequence (LCS) length via Hunt-Szymanski patience sorting.
    Runs in O((|A| + M) log |B|) where M is the number of matching pairs — sub-millisecond
    on road-network arc sequences where arc IDs rarely repeat.
    """
    if not seq_a or not seq_b:
        return 0

    positions_in_b: Dict[int, List[int]] = defaultdict(list)
    for idx, val in enumerate(seq_b):
        positions_in_b[val].append(idx)

    # Reverse index lists so each element of seq_a can match at most once
    for val in positions_in_b:
        positions_in_b[val].reverse()

    tails: List[int] = []
    for val in seq_a:
        pos_list = positions_in_b.get(val)
        if not pos_list:
            continue
        for pos in pos_list:
            ins = bisect_left(tails, pos)
            if ins == len(tails):
                tails.append(pos)
            else:
                tails[ins] = pos

    return len(tails)


def compute_sequence_metrics(
    predicted_obs_arcs: Sequence[int],
    reference_route: Sequence[int],
) -> Dict[str, float]:
    """
    Compute sequence-level route reconstruction quality comparing the compressed
    predicted traversed arc sequence against the ground-truth `.route` sequence:
      - route_lcs_precision / recall / f1: exact order-preserving LCS F1
      - route_set_jaccard: Jaccard overlap of unique traversed road arcs
      - route_length_ratio: len(predicted_traversed_route) / len(reference_route)
    """
    pred_route = compress_route(predicted_obs_arcs)
    ref_route = [int(x) for x in reference_route]

    n_pred = len(pred_route)
    n_ref = len(ref_route)

    if n_pred == 0 and n_ref == 0:
        return {
            "pred_route_len": 0,
            "ref_route_len": 0,
            "route_lcs_len": 0,
            "route_lcs_precision": 1.0,
            "route_lcs_recall": 1.0,
            "route_lcs_f1": 1.0,
            "route_set_jaccard": 1.0,
            "route_multiset_f1": 1.0,
            "route_length_ratio": 1.0,
        }

    lcs_len = _exact_lcs_length(pred_route, ref_route)
    lcs_prec = lcs_len / n_pred if n_pred > 0 else 0.0
    lcs_rec = lcs_len / n_ref if n_ref > 0 else 0.0
    lcs_f1 = (2.0 * lcs_prec * lcs_rec / (lcs_prec + lcs_rec)) if (lcs_prec + lcs_rec) > 0 else 0.0

    set_pred = set(pred_route)
    set_ref = set(ref_route)
    union_sz = len(set_pred | set_ref)
    jaccard = len(set_pred & set_ref) / union_sz if union_sz > 0 else 0.0

    # Multiset overlap F1
    c_pred = Counter(pred_route)
    c_ref = Counter(ref_route)
    inter_cnt = sum((c_pred & c_ref).values())
    m_f1 = (2.0 * inter_cnt / (n_pred + n_ref)) if (n_pred + n_ref) > 0 else 0.0

    return {
        "pred_route_len": n_pred,
        "ref_route_len": n_ref,
        "route_lcs_len": lcs_len,
        "route_lcs_precision": lcs_prec,
        "route_lcs_recall": lcs_rec,
        "route_lcs_f1": lcs_f1,
        "route_set_jaccard": jaccard,
        "route_multiset_f1": m_f1,
        "route_length_ratio": (n_pred / n_ref) if n_ref > 0 else 0.0,
    }
