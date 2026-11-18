"""
src/pipeline.py - Production Map-Matching Prediction Pipeline for PathSnap.

Coordinates the complete inference workflow:
1. Validate trip ID and load raw GPS observations, nodes, and arcs (via src.data_io).
2. Generate road segment candidates using spatial cKDTree indexing (via src.candidates).
3. Compute 26 leak-free spatial, heading, kinematic, and graph features (via src.features).
4. Score candidates with Person B's trained final model (LightGBM).
5. Decode the optimal continuous road sequence using Person A's HMM/Viterbi (via src.hmm).
6. Package the predicted route, traversed road sequence, and telemetry into a clean result.

CRITICAL DATA SEPARATION:
Ground-truth reference route (.route) is NEVER loaded or accessed by this pipeline.
"""

import os
import sys
import time

# Prevent joblib loky from calling external subprocess to count CPUs in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import itertools
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_io import load_trip, TripData
from src.candidates import generate_candidates_for_trip
from src.features import compute_candidate_features
from src.models import FEATURE_COLS
from src.hmm import ViterbiMapMatcher


@dataclass
class PipelineResult:
    """Structured container for end-to-end map matching prediction results."""
    trip_id: str
    num_observations: int
    predicted_arcs: List[int]
    traversed_route: List[int]
    num_arcs_traversed: int
    probabilities: List[float]
    continuity_pct: float
    processing_time_ms: float
    model_name: str
    decoder: str
    candidates_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a clean dictionary suitable for API and JSON serialization."""
        return {
            "trip_id": self.trip_id,
            "num_observations": self.num_observations,
            "predicted_arcs": self.predicted_arcs,
            "traversed_route": self.traversed_route,
            "num_arcs_traversed": self.num_arcs_traversed,
            "probabilities": [round(p, 4) for p in self.probabilities],
            "continuity_pct": round(self.continuity_pct, 2),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_name": self.model_name,
            "decoder": self.decoder,
        }


class MapMatchingPipeline:
    """
    Production-grade inference pipeline coordinating GPS processing,
    candidate generation, feature extraction, ML scoring, and HMM decoding.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        K: int = 10,
        viterbi_matcher: Optional[ViterbiMapMatcher] = None,
    ):
        self.K = K
        self.model_path = model_path or os.path.join(PROJECT_ROOT, "artifacts", "lightgbm.joblib")
        self.model = None
        self.model_name = "LightGBM"
        self.viterbi_matcher = viterbi_matcher or ViterbiMapMatcher()
        self._load_model()

    def _load_model(self):
        """Load and cache the trained ML model artifact."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Ensure Person B's trained model exists in artifacts/."
            )
        self.model = joblib.load(self.model_path)
        if hasattr(self.model, "name"):
            self.model_name = self.model.name
        elif hasattr(self.model, "model_name"):
            self.model_name = self.model.model_name
        else:
            self.model_name = "LightGBM"

    def is_ready(self) -> bool:
        """Check whether the model and decoder components are fully initialized."""
        return self.model is not None and self.viterbi_matcher is not None

    def predict_trip(
        self,
        trip_id: str,
        data_root: str = "Student folder",
        partition: str = "student_test",
        return_candidates_df: bool = False,
    ) -> PipelineResult:
        """
        Execute full map-matching prediction for a single trip.
        
        Args:
            trip_id: Target trip directory name (e.g. '00000052')
            data_root: Root dataset folder
            partition: 'student_test', 'train', or custom directory
            return_candidates_df: Whether to retain candidate feature DataFrame
            
        Returns:
            PipelineResult with predicted arc sequences, continuity, and execution telemetry.
        """
        if not self.is_ready():
            raise RuntimeError("Pipeline components are not loaded.")

        # 1. Resolve trip directory
        trip_dir = os.path.join(PROJECT_ROOT, data_root, partition, trip_id)
        if not os.path.isdir(trip_dir):
            # Check relative to current working directory
            alt_dir = os.path.join(data_root, partition, trip_id)
            if os.path.isdir(alt_dir):
                trip_dir = alt_dir
            else:
                raise FileNotFoundError(
                    f"Trip '{trip_id}' not found in partition '{partition}' at {trip_dir}"
                )

        t_start = time.perf_counter()

        # 2. Load raw trip observations (NEVER uses trip.route)
        trip: TripData = load_trip(trip_dir)
        num_obs = len(trip.track)
        if num_obs == 0:
            raise ValueError(f"Trip '{trip_id}' has zero GPS observations in .track file.")

        # 3. Generate candidate road arcs via cKDTree
        cand_df, _ = generate_candidates_for_trip(trip, K=self.K)

        # 4. Compute 26 leak-free features
        features_df = compute_candidate_features(trip, cand_df)

        # 5. Model scoring (LightGBM candidate probabilities)
        X = features_df[FEATURE_COLS].to_numpy()
        probs = self.model.predict_proba(X)
        features_df["prob"] = probs

        # 6. HMM / Viterbi sequence decoding (Person A's decoder)
        arcs_array = trip.arcs.to_numpy()
        decoded_df = self.viterbi_matcher.decode_trip(
            obs_candidates=features_df[["obs_idx", "arc_id", "prob"]],
            arcs=arcs_array,
        )

        predicted_arcs: List[int] = decoded_df["viterbi_arc_id"].tolist()
        probabilities: List[float] = decoded_df["viterbi_prob"].tolist()

        # 7. Reconstruct traversed road sequence (compress consecutive duplicate runs)
        traversed_route = [k for k, _ in itertools.groupby(predicted_arcs)]

        # 8. Compute route continuity (% valid transitions in predicted sequence)
        continuity_pct = self._calculate_route_continuity(predicted_arcs, arcs_array)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return PipelineResult(
            trip_id=trip_id,
            num_observations=num_obs,
            predicted_arcs=predicted_arcs,
            traversed_route=traversed_route,
            num_arcs_traversed=len(traversed_route),
            probabilities=probabilities,
            continuity_pct=continuity_pct,
            processing_time_ms=t_elapsed_ms,
            model_name=self.model_name,
            decoder="HMM Viterbi",
            candidates_df=features_df if return_candidates_df else None,
        )

    @staticmethod
    def _calculate_route_continuity(predicted_arcs: List[int], arcs_array: np.ndarray) -> float:
        """
        Calculate route continuity as percentage of consecutive transitions that are
        either identical or directly connected in the road network graph.
        """
        if len(predicted_arcs) <= 1:
            return 100.0

        curr_arcs = np.array(predicted_arcs[:-1], dtype=np.int32)
        next_arcs = np.array(predicted_arcs[1:], dtype=np.int32)

        # Check bounds
        num_arcs = len(arcs_array)
        valid_mask = (curr_arcs < num_arcs) & (next_arcs < num_arcs)
        if not np.all(valid_mask):
            return 0.0

        from_nodes = arcs_array[:, 0]
        to_nodes = arcs_array[:, 1]

        is_same = (curr_arcs == next_arcs)
        is_connected = (to_nodes[curr_arcs] == from_nodes[next_arcs])
        is_valid = is_same | is_connected

        return float(np.mean(is_valid) * 100.0)


# Module-level singleton pipeline instance for fast reuse in API and scripts
_GLOBAL_PIPELINE: Optional[MapMatchingPipeline] = None


def get_pipeline(model_path: Optional[str] = None) -> MapMatchingPipeline:
    """Return the cached global MapMatchingPipeline singleton, initializing it once if needed."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = MapMatchingPipeline(model_path=model_path)
    return _GLOBAL_PIPELINE
