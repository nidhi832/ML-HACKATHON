# PathSnap & TRACE.AI: Robust Machine Learning Map Matching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PathSnap & TRACE.AI** is a high-accuracy, leak-free machine learning map-matching platform designed for snapping noisy GPS trajectories onto complex, directed road networks. Built for the **PES University ML Hackathon (UE24CS352A)**.

---

## 🚀 Key Highlights & Verified Results

- **Global Spatial Invariance:** Automatically detects trip centroids and projects coordinates to local metric UTM grids (`pyproj`), ensuring flat metric Euclidean distance computations.
- **Ultra-Fast Candidate Recall:** Vectorized $k$-d trees (`scipy.spatial.cKDTree`) achieve **98.83% candidate recall** ($K=10$, candidate miss rate 1.17%).
- **26 Leak-Free Features:** Perpendicular distance, projection fraction, directed heading differences ($\Delta\theta, \cos\Delta\theta$), turn angle, acceleration, and network connectivity (degrees, reverse arc flags). Reference `.route` is **never** accessed during prediction.
- **Multi-Family Benchmarking (5-Fold GroupKFold by Trip):**
  - **Random Forest:** CV Macro F1: **0.9042 ± 0.0360** | Test Macro F1: **0.9237** | Log Loss: **0.0659**
  - **LightGBM (Champion):** CV Macro F1: **0.9008 ± 0.0435** | Test Top-1 Acc: **87.16%** | Test Macro F1: **0.9293** | Log Loss: **0.0613**
  - **Deep MLP (PyTorch):** CV Macro F1: **0.8999 ± 0.0369** | Test Macro F1: **0.9224** | Log Loss: **0.0700**
  - **Logistic Regression:** CV Macro F1: **0.8743 ± 0.0301** | Test Macro F1: **0.8989** | Log Loss: **0.0991**
  - **Subsampled KNN:** CV Macro F1: **0.8551 ± 0.0400** | Test Macro F1: **0.8742** | Log Loss: **0.1684**
  - **Gaussian Naive Bayes:** CV Macro F1: **0.6671 ± 0.0311** | Test Macro F1: **0.6583** | Log Loss: **1.4502**
- **Hidden Markov Model (HMM) Viterbi Sequence Reconstruction (Person A):**
  - **Observation Top-1 Accuracy:** **87.16% $\to$ 92.87%** (pooled across 19,160 observations, **+5.71% gain**).
  - **Per-Trip Mean Top-1 Accuracy:** **93.29%** across the 10 test trips.
  - **Physical Route Continuity:** **93.93% $\to$ 98.96%** (+5.03%).
  - **Reverse 180° U-Turns:** **779 $\to$ 3** (99.6% false U-turns eliminated!).
  - **Disconnected Jumps:** **685 $\to$ 23** (662 invalid transitions removed).
  - **Route LCS F1:** **0.7511 $\to$ 0.9502** (+0.1991).
  - **Gap & Dropout Resilience:** Dynamically soft-scales transition weights when $\Delta t > 30\text{ s}$ or displacement $> 250\text{ m}$.
- **Bayesian Hyperparameter Optimization (Person B):**
  - Optuna TPE (seed 42) across GroupKFold on 80 training trips improves CV Macro F1 from **0.9130 to 0.9170** with reduced fold variance (std dropped from 0.0192 to 0.0157).
  - Champion model serialized in `artifacts/final_model.joblib`.
- **Zero Fabricated / Cached Metrics (Person C):**
  - All predictions execute live via `get_pipeline().predict_trip()`.
  - Zero hardcoded additions; real scoring computed server-side via `src/scoring.py` strictly when `.route` is provided.

---

## 📁 Repository Structure

```
.
├── api/                   # FastAPI backend service (endpoints, inference, SPA hosting)
│   └── main.py
├── ui/                    # TRACE.AI React + Tailwind + Leaflet frontend application
│   ├── src/               # Geospatial components, map playback, model comparison, settings
│   ├── dist/              # Built production assets served by FastAPI
│   └── package.json
├── configs/               # System & Optuna tuning configurations
├── src/                   # Core production source code
│   ├── data_io.py         # NamedTuple TripData, parquet and raw loaders
│   ├── geo.py             # Local UTM projection, orthogonal point-to-segment distance
│   ├── candidates.py      # Vectorized cKDTree candidate generator (K=10)
│   ├── labels.py          # Monotonic Dynamic Programming alignment from .route
│   ├── features.py        # 26 leak-free spatial, heading, kinematic, and graph features
│   ├── models.py          # 6 model family wrappers (Sklearn, LightGBM, PyTorch)
│   ├── evaluate.py        # Top-1 accuracy, candidate metrics, route continuity
│   ├── hmm.py             # Hidden Markov Model & Viterbi sequence decoder (Person A)
│   ├── pipeline.py        # Production inference pipeline (leak-free)
│   └── scoring.py         # Server-side evaluation against reference routes
├── scripts/               # Pipeline execution & benchmark scripts
│   ├── 01_eda.py          # Exploratory data analysis, quality checks, distributions
│   ├── 02_build_dataset.py# Feature extraction & parquet dataset generation
│   ├── 03_train_compare.py# 5-Fold GroupKFold training, evaluation & artifact saving
│   ├── 04_plots.py        # Generates publication figures to results/figures/
│   ├── 05_hmm_eval.py     # HMM evaluation & ablation benchmarks (Person A)
│   ├── 06_folium_maps.py  # Generates multi-layer interactive Folium HTML maps
│   ├── 07_bayes_opt.py    # Optuna TPE Bayesian optimization (Person B)
│   ├── test_all_trips.py  # End-to-end batch test across student-test trips
│   ├── pre_evaluation_check.py # Automated pre-evaluation checklist
│   └── build_report_pdf.py# Generates publication PDF report
├── artifacts/             # Serialized model weights (.joblib, .pt)
├── results/               # Experiment logs, CSV metrics, and figures
├── maps/                  # Interactive Folium HTML maps (Best, Typical, Worst)
├── report/                # Technical report markdown and PDF deliverable
├── DEPLOY.md              # Distributed Two-Laptop deployment guide
├── launch_trace_ai.bat    # Windows 1-click launcher
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🌐 Launch TRACE.AI Web Dashboard

```bash
# Double-click 'launch_trace_ai.bat' on Windows or run:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** in your browser:
- **Overview Dashboard:** Live KPI telemetry, interactive road network map, confidence distributions.
- **GPS Map Matching:** Model selector (6 models), K candidate slider, HMM toggle, interactive map with layer toggles, and **live route trajectory playback (1x/2x/4x)**!
- **Trip Explorer:** Sortable, searchable data table across all trajectories with accuracy ratings.
- **Model Performance:** Recharts bar and line charts comparing all 6 models, CV GroupKFold folds, and PyTorch epoch loss convergence.
- **Evaluation Dashboard:** Official benchmark analysis across 10 evaluation trips with CSV export.
- **Settings:** Configurable backend API Server URL for multi-laptop deployment with live ping test.
- **API Documentation:** Interactive Swagger UI at **http://localhost:8000/docs**.

---

## 💻 Two-Laptop Distributed Setup (Evaluation Mode)

Detailed step-by-step instructions are available in [DEPLOY.md](file:///d:/Downloads/ml%20hack/DEPLOY.md).

1. **Laptop 1 (Inference Server):**
   ```powershell
   # 1. Open Windows Firewall for port 8000
   New-NetFirewallRule -DisplayName "TRACE_AI_API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

   # 2. Set evaluation directory (optional, defaults to Student folder/student_test)
   $env:EVAL_DATA_DIR = "Student folder\student_test"

   # 3. Start server
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Laptop 2 (Evaluator Client):**
   ```powershell
   # 1. Verify connection
   curl http://<LAPTOP_1_IP>:8000/health

   # 2. Run the automated pre-evaluation checklist
   python scripts/pre_evaluation_check.py --url http://<LAPTOP_1_IP>:8000
   ```

---

## ⚡ Execution Pipeline

### 1. Run Pre-Evaluation Checklist
```bash
python scripts/pre_evaluation_check.py
```
Validates `/health`, `/trips`, and `/predict` across all 10 trips with a live PASS/FAIL table.

### 2. Run Batch Test Across All 10 Trips
```bash
python scripts/test_all_trips.py
```
Evaluates all 10 student-test trips end-to-end and outputs results to `results/student_test_all_trips.json`.

### 3. Run Optuna Bayesian Optimization (Person B)
```bash
python scripts/07_bayes_opt.py
```
Executes Optuna TPE optimization across GroupKFold splits, generates the optimization history plot, and compares baseline vs tuned models.

### 4. Run HMM Evaluation & Ablation (Person A)
```bash
python scripts/05_hmm_eval.py
```
Benchmarks Classifier-Only vs Classifier + HMM Viterbi across all 6 model families and generates ablation tables and Folium maps.

### 5. Generate Technical Report PDF
```bash
python scripts/build_report_pdf.py
```
Generates `report/technical_report.pdf` from `report/technical_report.md`.
