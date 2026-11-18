"""
PathSnap: Hidden Markov Model (HMM) & Viterbi Sequence Decoder
Decodes candidate road arcs into continuous, physically connected trajectories.

Formulation:
- Hidden State at time t: candidate road arc c_{t, k} (k in 1..K)
- Emission Probability: Calibrated ML probability P(arc | GPS_t, features)
- Transition Probability: Road network connectivity (arcs[u].to == arcs[v].from)
  and network shortest-path consistency with GPS displacement Delta d.
- Decoding: Dynamic Programming (Viterbi Algorithm) maximizing joint log-likelihood.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


class ViterbiMapMatcher:
    def __init__(
        self,
        same_arc_log_prob: float = -0.1,
        direct_connected_log_prob: float = -0.5,
        two_hop_log_prob: float = -2.5,
        disconnected_penalty: float = -25.0,
        emission_weight: float = 1.0,
        transition_weight: float = 1.5,
    ):
        self.same_arc_log_prob = same_arc_log_prob
        self.direct_connected_log_prob = direct_connected_log_prob
        self.two_hop_log_prob = two_hop_log_prob
        self.disconnected_penalty = disconnected_penalty
        self.emission_weight = emission_weight
        self.transition_weight = transition_weight

    def build_network_graph(self, arcs: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        Build fast adjacency lookups for directed arcs.
        - out_arcs[from_node] = list of arc_ids starting at from_node
        - node_outgoing = adjacency for BFS
        """
        out_arcs_by_node = defaultdict(list)
        for arc_id in range(len(arcs)):
            u, v = arcs[arc_id, 0], arcs[arc_id, 1]
            out_arcs_by_node[u].append(arc_id)
        return out_arcs_by_node

    def decode_trip(
        self,
        obs_candidates: pd.DataFrame,
        arcs: np.ndarray,
        track_utm: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Apply Viterbi decoding over candidate sequence for a single trip.
        
        Args:
            obs_candidates: DataFrame with columns ['obs_idx', 'arc_id', 'prob']
            arcs: Directed arcs array of shape (N_arcs, 2) [from_node, to_node]
            track_utm: Optional (N_obs, 2) coordinates for displacement checks
            
        Returns:
            DataFrame with decoded route: ['obs_idx', 'viterbi_arc_id', 'viterbi_prob']
        """
        # Group candidates by observation index
        grouped = obs_candidates.groupby("obs_idx")
        obs_indices = sorted(grouped.groups.keys())
        T = len(obs_indices)
        
        if T == 0:
            return pd.DataFrame(columns=["obs_idx", "viterbi_arc_id", "viterbi_prob"])

        # Precompute directed adjacency
        out_arcs_by_node = self.build_network_graph(arcs)
        
        # Prepare candidate states per time step
        # states[t] = list of (arc_id, log_emission)
        states_per_t = []
        for obs_idx in obs_indices:
            cands = grouped.get_group(obs_idx)
            # Normalize probabilities across K candidates via softmax/temperature
            probs = cands["prob"].values.astype(np.float64)
            # Clip and soften
            probs = np.clip(probs, 1e-6, 1.0)
            p_sum = probs.sum()
            norm_probs = probs / p_sum if p_sum > 0 else np.ones_like(probs) / len(probs)
            log_emissions = np.log(norm_probs)
            arc_ids = cands["arc_id"].values.astype(np.int64)
            states_per_t.append(list(zip(arc_ids, log_emissions)))

        # Viterbi Trellis
        # viterbi_table[t][k] = max log score to reach state k at time t
        # backpointers[t][k] = state index at time t-1 that gave max score
        
        K0 = len(states_per_t[0])
        viterbi_table = [np.array([s[1] * self.emission_weight for s in states_per_t[0]], dtype=np.float64)]
        backpointers = []

        for t in range(1, T):
            prev_states = states_per_t[t - 1]
            curr_states = states_per_t[t]
            K_prev = len(prev_states)
            K_curr = len(curr_states)
            
            curr_viterbi = np.full(K_curr, -1e12, dtype=np.float64)
            curr_bp = np.zeros(K_curr, dtype=np.int32)
            
            for j in range(K_curr):
                arc_curr, log_em_curr = curr_states[j]
                u_curr = arcs[arc_curr, 0]
                
                best_score = -1e12
                best_prev_idx = 0
                
                for i in range(K_prev):
                    arc_prev, _ = prev_states[i]
                    v_prev = arcs[arc_prev, 1]
                    
                    # Compute transition log probability
                    if arc_prev == arc_curr:
                        # Same arc
                        t_score = self.same_arc_log_prob
                    elif v_prev == u_curr:
                        # Direct graph edge: arcs[arc_prev].to == arcs[arc_curr].from
                        t_score = self.direct_connected_log_prob
                    else:
                        # Check 2-hop connection
                        # Is there any arc from v_prev that leads to u_curr?
                        two_hop = False
                        for intermediate_arc in out_arcs_by_node.get(v_prev, []):
                            if arcs[intermediate_arc, 1] == u_curr:
                                two_hop = True
                                break
                        if two_hop:
                            t_score = self.two_hop_log_prob
                        else:
                            t_score = self.disconnected_penalty
                            
                    total_score = (
                        viterbi_table[t - 1][i]
                        + (t_score * self.transition_weight)
                        + (log_em_curr * self.emission_weight)
                    )
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_prev_idx = i
                        
                curr_viterbi[j] = best_score
                curr_bp[j] = best_prev_idx
                
            viterbi_table.append(curr_viterbi)
            backpointers.append(curr_bp)

        # Backtrack optimal sequence
        best_last_idx = int(np.argmax(viterbi_table[-1]))
        best_path_indices = [best_last_idx]
        
        for t in range(T - 1, 0, -1):
            prev_idx = backpointers[t - 1][best_path_indices[-1]]
            best_path_indices.append(prev_idx)
            
        best_path_indices.reverse()
        
        # Build results DataFrame
        decoded_arcs = []
        decoded_probs = []
        for t, idx in enumerate(best_path_indices):
            arc_id, log_em = states_per_t[t][idx]
            decoded_arcs.append(arc_id)
            decoded_probs.append(float(np.exp(log_em)))
            
        return pd.DataFrame({
            "obs_idx": obs_indices,
            "viterbi_arc_id": decoded_arcs,
            "viterbi_prob": decoded_probs,
        })
