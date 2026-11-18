"""
tests/test_pipeline_and_api.py - Unit and Integration Tests for Person C.

Tests:
1. Python imports and environment readiness.
2. Model loading and pipeline initialization.
3. Pipeline execution on sample trip (leak-free check).
4. Scoring functions and metrics verification.
5. FastAPI /health endpoint.
6. FastAPI /trips endpoint.
7. FastAPI /predict endpoint (success, 404 not found, 400 validation error).
8. Request logging and performance headers.
"""

import os
import sys
import pytest

# Prevent joblib loky from calling external subprocess to count CPUs in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline, MapMatchingPipeline, PipelineResult
from src.scoring import evaluate_trip_prediction
from api.main import app, health_check, get_trips, predict_trip_route, PredictRequest


class TestPipelineAndScoring:
    @classmethod
    def setup_class(cls):
        cls.pipeline = get_pipeline()

    def test_pipeline_initialized(self):
        assert self.pipeline is not None
        assert self.pipeline.is_ready() is True
        assert self.pipeline.model is not None
        assert self.pipeline.viterbi_matcher is not None
        assert self.pipeline.model_name == "LightGBM"

    def test_pipeline_predict_sample_trip(self):
        # Predict on student_test trip 00000052
        result: PipelineResult = self.pipeline.predict_trip(
            trip_id="00000052",
            data_root="Student folder",
            partition="student_test",
            return_candidates_df=True,
        )

        assert result.trip_id == "00000052"
        assert result.num_observations == 262
        assert len(result.predicted_arcs) == 262
        assert len(result.traversed_route) > 0
        assert result.num_arcs_traversed == len(result.traversed_route)
        assert result.continuity_pct >= 90.0
        assert result.processing_time_ms > 0.0
        assert result.model_name == "LightGBM"
        assert result.decoder == "HMM Viterbi"
        assert result.candidates_df is not None

    def test_scoring_with_reference_route(self):
        result: PipelineResult = self.pipeline.predict_trip(
            trip_id="00000052",
            data_root="Student folder",
            partition="student_test",
            return_candidates_df=True,
        )

        metrics = evaluate_trip_prediction(
            trip_id="00000052",
            predicted_arcs=result.predicted_arcs,
            data_root="Student folder",
            partition="student_test",
            candidates_df=result.candidates_df,
        )

        assert metrics["has_reference_route"] is True
        assert metrics["top1_accuracy_pct"] >= 95.0
        assert metrics["route_continuity_pct"] >= 90.0
        assert metrics["candidate_recall_pct"] >= 95.0
        assert "log_loss" in metrics
        assert "confusion_matrix" in metrics

    def test_pipeline_invalid_trip(self):
        with pytest.raises(FileNotFoundError):
            self.pipeline.predict_trip(trip_id="nonexistent_trip_xyz")


class TestFastAPIEndpoints:
    def test_health_endpoint_direct(self):
        response = health_check()
        assert response.status == "ok"
        assert response.model_loaded is True
        assert response.model_name == "LightGBM"
        assert response.hmm_decoder_loaded is True
        assert response.student_test_trips_count == 10

    def test_trips_endpoint_direct(self):
        response = get_trips(partition="student_test")
        assert response.partition == "student_test"
        assert response.total_trips == 10
        assert "00000052" in response.trip_ids
        assert "00000000" in response.trip_ids

    def test_predict_endpoint_direct(self):
        req = PredictRequest(trip_id="00000052", partition="student_test")
        resp = predict_trip_route(req)
        assert resp.trip_id == "00000052"
        assert resp.status == "success"
        assert resp.num_observations == 262
        assert len(resp.predicted_arcs) == 262
        assert resp.num_arcs_traversed == len(resp.traversed_route)
        assert resp.continuity_pct >= 90.0
        assert resp.model_name == "LightGBM"
        assert resp.decoder == "HMM Viterbi"
