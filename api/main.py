"""
api/main.py - Production FastAPI backend for PathSnap Map-Matching System.

Provides REST endpoints for distributed two-laptop hackathon deployment:
- GET  /health  : System health check, model readiness, and dataset verification.
- GET  /trips   : Inventory of evaluation trips available (partition selectable).
- POST /predict : Execute end-to-end map-matching prediction for a selected trip.

Key Features:
- CORS enabled for cross-origin access from Laptop 2 (UI client).
- Structured request logging middleware with latency and status tracking.
- Pydantic v2 request and response validation schemas.
- Complete data separation: NEVER returns or leaks reference routes (.route).
- Configurable host/port listening on 0.0.0.0 by default.
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
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline, MapMatchingPipeline, PipelineResult
from src.data_io import list_trips

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pathsnap.api")

# Application lifecycle: load pipeline singleton on startup
pipeline: Optional[MapMatchingPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring model and decoder are loaded on startup."""
    global pipeline
    logger.info("Initializing PathSnap prediction pipeline...")
    try:
        pipeline = get_pipeline()
        logger.info(
            f"Pipeline successfully initialized with model '{pipeline.model_name}' "
            f"and HMM Viterbi decoder."
        )
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline on startup: {e}")
        pipeline = None
    yield
    logger.info("Shutting down PathSnap API service.")


app = FastAPI(
    title="PathSnap Map-Matching API",
    description=(
        "Production REST API for snapping noisy GPS trajectories to road networks. "
        "Built for PES University UE24CS352A ML Hackathon."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# -------------------------------------------------------------------------
# CORS Middleware: Critical for Two-Laptop execution (Laptop 1 API -> Laptop 2 UI)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, enabling LAN cross-laptop access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Request Logging Middleware: Tracks timestamp, IP, method, path, and duration
# -------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url_path = request.url.path

    try:
        response: Response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"{method} {url_path} - Status: {response.status_code} - "
            f"IP: {client_ip} - Time: {process_time_ms:.2f}ms"
        )
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response
    except Exception as exc:
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"{method} {url_path} - ERROR: {exc} - "
            f"IP: {client_ip} - Time: {process_time_ms:.2f}ms"
        )
        raise exc


# -------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when ready, 'degraded' if components missing")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded in memory")
    model_name: str = Field(..., description="Name of the loaded ML model")
    hmm_decoder_loaded: bool = Field(..., description="Whether the HMM/Viterbi decoder is ready")
    student_test_trips_count: int = Field(..., description="Number of student-testing trips available")
    available_partitions: List[str] = Field(..., description="List of recognized dataset partitions")


class TripsResponse(BaseModel):
    partition: str = Field(..., description="Dataset partition queried")
    total_trips: int = Field(..., description="Total available trips in this partition")
    trip_ids: List[str] = Field(..., description="List of valid trip ID strings")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="ID of the trip to predict (e.g. '00000052')", examples=["00000052"])
    partition: Optional[str] = Field(
        default="student_test",
        description="Dataset partition ('student_test' or 'train')",
        examples=["student_test"],
    )


class PredictResponse(BaseModel):
    trip_id: str = Field(..., description="Trip identifier")
    status: str = Field(default="success", description="Prediction status")
    num_observations: int = Field(..., description="Total GPS observation points processed")
    predicted_arcs: List[int] = Field(..., description="Decoded road arc ID for each GPS point")
    traversed_route: List[int] = Field(..., description="Sequence of road arcs traversed (duplicates condensed)")
    num_arcs_traversed: int = Field(..., description="Count of traversed road segments")
    probabilities: List[float] = Field(..., description="HMM confidence probability per observation")
    continuity_pct: float = Field(..., description="Percentage of connected consecutive road transitions")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    model_name: str = Field(..., description="ML model used for candidate scoring")
    decoder: str = Field(..., description="Sequential decoder used (HMM Viterbi)")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    tags=["System"],
)
def health_check():
    """
    Check API operational health, model loading status, and dataset availability.
    Returns HTTP 200 if ready, HTTP 503 if pipeline components are unavailable.
    """
    global pipeline
    if pipeline is None:
        try:
            pipeline = get_pipeline()
        except Exception as e:
            logger.warning(f"Health check: Pipeline could not be initialized: {e}")

    model_loaded = pipeline is not None and pipeline.model is not None
    hmm_loaded = pipeline is not None and pipeline.viterbi_matcher is not None
    model_name = pipeline.model_name if pipeline else "None"

    # Count available test trips
    test_trips = []
    try:
        test_trips = list_trips(os.path.join(PROJECT_ROOT, "Student folder"), "student_test")
    except Exception:
        try:
            test_trips = list_trips("Student folder", "student_test")
        except Exception:
            pass

    partitions = []
    student_root = os.path.join(PROJECT_ROOT, "Student folder")
    if os.path.isdir(student_root):
        partitions = [d for d in os.listdir(student_root) if os.path.isdir(os.path.join(student_root, d))]

    is_healthy = model_loaded and hmm_loaded and len(test_trips) > 0
    resp_status = "ok" if is_healthy else "degraded"

    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    response_data = HealthResponse(
        status=resp_status,
        model_loaded=model_loaded,
        model_name=model_name,
        hmm_decoder_loaded=hmm_loaded,
        student_test_trips_count=len(test_trips),
        available_partitions=partitions,
    )

    if not is_healthy:
        return JSONResponse(status_code=status_code, content=response_data.model_dump())
    return response_data


@app.get(
    "/trips",
    response_model=TripsResponse,
    summary="List Available Trips",
    tags=["Data"],
)
def get_trips(
    partition: str = Query(
        default="student_test",
        description="Dataset partition ('student_test' or 'train')",
    )
):
    """
    Retrieve valid trip IDs for the chosen partition.
    Does NOT leak ground-truth reference routes.
    """
    valid_partitions = ["student_test", "train"]
    if partition not in valid_partitions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid partition '{partition}'. Must be one of {valid_partitions}.",
        )

    data_root = os.path.join(PROJECT_ROOT, "Student folder")
    if not os.path.isdir(data_root):
        data_root = "Student folder"

    try:
        trips = list_trips(data_root, partition)
        trip_ids = [tid for tid, _ in trips]
        return TripsResponse(
            partition=partition,
            total_trips=len(trip_ids),
            trip_ids=trip_ids,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing trips for partition '{partition}': {e}",
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Map-Matched Route",
    tags=["Inference"],
)
def predict_trip_route(request: PredictRequest):
    """
    Execute full map-matching prediction for a selected trip ID.
    
    1. Validates trip existence.
    2. Loads GPS observations, road nodes, and directed arcs.
    3. Generates road candidates (cKDTree).
    4. Computes 26 leak-free features.
    5. Scores candidates with trained LightGBM model.
    6. Decodes optimal sequence via HMM / Viterbi.
    7. Returns predicted route and sequence telemetry.
    
    CRITICAL: Never exposes or relies upon ground-truth reference routes.
    """
    global pipeline
    if pipeline is None:
        try:
            pipeline = get_pipeline()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Prediction pipeline could not be loaded: {e}",
            )

    trip_id = request.trip_id.strip()
    partition = (request.partition or "student_test").strip()

    if not trip_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'trip_id' must not be empty.",
        )

    try:
        result: PipelineResult = pipeline.predict_trip(
            trip_id=trip_id,
            partition=partition,
            return_candidates_df=False,
        )

        return PredictResponse(
            trip_id=result.trip_id,
            status="success",
            num_observations=result.num_observations,
            predicted_arcs=result.predicted_arcs,
            traversed_route=result.traversed_route,
            num_arcs_traversed=result.num_arcs_traversed,
            probabilities=[round(p, 4) for p in result.probabilities],
            continuity_pct=round(result.continuity_pct, 2),
            processing_time_ms=round(result.processing_time_ms, 2),
            model_name=result.model_name,
            decoder=result.decoder,
        )

    except FileNotFoundError as e:
        logger.warning(f"Trip not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip '{trip_id}' not found in partition '{partition}'.",
        )
    except ValueError as e:
        logger.warning(f"Invalid input data for trip '{trip_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected prediction error for trip '{trip_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal prediction pipeline failure: {str(e)}",
        )


# -------------------------------------------------------------------------
# Standalone CLI entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("PATHSNAP_HOST", "0.0.0.0")
    port = int(os.environ.get("PATHSNAP_PORT", 8000))
    logger.info(f"Starting PathSnap API server on http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
