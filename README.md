# PathSnap: Robust Machine Learning Map Matching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PathSnap** is a high-accuracy, leak-free machine learning system for snapping noisy GPS trajectories onto complex, directed road networks. Built for the **PES University ML Hackathon (UE24CS352A)**.

---

## 🚀 Key Highlights & Results

- **Global Spatial Invariance:** Automatically detects trip centroids and projects coordinates to local metric UTM grids (`pyproj`), ensuring flat metric Euclidean distance computations.
- **Ultra-Fast Candidate Recall:** Vectorized $k$-d trees (`scipy.spatial.cKDTree`) achieve **98.83% candidate recall** ($K=10$, candidate miss rate 1.17%).
- **26 Leak-Free Features:** Incorporates perpendicular distance, projection fraction, directed heading differences ($\Delta\theta, \cos\Delta\theta$), turn angle, acceleration, and network connectivity (degrees, reverse arc flags).
- **Multi-Family Benchmarking:** Evaluated 6 diverse ML families using **5-Fold GroupKFold by trip** (never splitting GPS points of a trip across train and validation):
  - **Random Forest:** CV Macro F1: **0.9042 ± 0.0360** | Test Macro F1: **0.9237** | Log Loss: **0.0659**
  - **LightGBM:** CV Macro F1: **0.9008 ± 0.0435** | Test Top-1 Acc: **87.16%** | Test Macro F1: **0.9293** | Log Loss: **0.0613**
  - **Deep MLP (PyTorch):** CV Macro F1: **0.8999 ± 0.0369** | Test Macro F1: **0.9224** | Log Loss: **0.0700**
  - **Logistic Regression:** CV Macro F1: **0.8743 ± 0.0301** | Test Macro F1: **0.8989**
  - **Subsampled KNN:** CV Macro F1: **0.8551 ± 0.0400** | Test Macro F1: **0.8742**
  - **Gaussian Naive Bayes:** CV Macro F1: **0.6671 ± 0.0311**
- **Sequence Reconstruction:** Hidden Markov Model (HMM) Viterbi decoder in `src/hmm.py` enforcing directed graph connectivity and physical transition probabilities.

---

## 📁 Repository Structure

```
.
├── api/                   # FastAPI backend service (endpoints, inference, SPA serving)
│   └── main.py
├── ui/                    # TRACE.AI React + Tailwind + Leaflet frontend application
│   ├── src/
│   │   ├── components/    # InteractiveMap, RoutePlayback, MetricCard, Sidebar, Topbar
│   │   ├── pages/         # Dashboard, MapMatching, TripExplorer, ModelComparison, Evaluation, Settings
│   │   └── services/      # REST API client with offline fallback
│   ├── dist/              # Production-built bundle served by FastAPI
│   └── package.json
├── configs/               # System configurations and parameters
├── src/                   # Core production source code
│   ├── data_io.py         # NamedTuple TripData, parquet and raw loaders
│   ├── geo.py             # Local UTM projection, orthogonal point-to-segment distance
│   ├── candidates.py      # Vectorized cKDTree candidate generator (K=10)
│   ├── labels.py          # Monotonic Dynamic Programming alignment from .route
│   ├── features.py        # 26 leak-free spatial, heading, kinematic, and graph features
│   ├── models.py          # 6 model family wrappers (Sklearn, LightGBM, PyTorch)
│   ├── evaluate.py        # Top-1 accuracy, candidate metrics, route continuity
│   └── hmm.py             # Hidden Markov Model & Viterbi sequence decoder
├── scripts/               # Pipeline execution scripts
│   ├── 01_eda.py          # Exploratory data analysis, quality checks, distributions
│   ├── 02_build_dataset.py# Feature extraction & parquet dataset generation
│   ├── 03_train_compare.py# 5-Fold GroupKFold training, evaluation & artifact saving
│   ├── 04_plots.py        # Generates publication figures to results/figures/
│   ├── 06_folium_maps.py  # Generates multi-layer interactive Folium HTML maps
│   ├── demo.py            # Live interactive evaluator CLI demonstration
│   └── launch_trace_ai.py # One-command launcher for TRACE.AI web platform
├── artifacts/             # Serialized model weights (.joblib, .pt)
├── results/               # Experiment logs, CSV metrics, and figures
├── maps/                  # Interactive Folium HTML maps (Best, Typical, Worst)
├── report/                # Technical report source and documentation
├── launch_trace_ai.bat    # Windows 1-click launcher
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🌐 Launch TRACE.AI Web Dashboard

To launch the full interactive web application:

```bash
# One-click command:
python scripts/launch_trace_ai.py
# Or double-click 'launch_trace_ai.bat' on Windows
```
The application will start at **http://localhost:8000** and automatically open in your default browser:
- **Overview Dashboard:** Live KPI telemetry, interactive road network map, confidence distributions.
- **GPS Map Matching:** Model selector (6 models), K candidate slider, HMM toggle, interactive map with layer toggles, and **live route trajectory playback (1x/2x/4x)**!
- **Trip Explorer:** Sortable, searchable data table across all trajectories with accuracy ratings.
- **Model Performance:** Recharts bar and line charts comparing all 6 models, CV GroupKFold folds, and PyTorch epoch loss convergence.
- **Evaluation Dashboard:** Official benchmark analysis across 10 evaluation trips with CSV export.
- **API Documentation:** Interactive Swagger UI at **http://localhost:8000/docs**.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nidhi832/ML-HACKATHON.git
   cd ML-HACKATHON
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Execution Pipeline

### 1. Run Interactive Review Demo
```bash
python scripts/demo.py
```
This prints the consolidated multi-family benchmark table, displays candidate recall, ranks test trips, and automatically launches the multi-layer Folium web map.

### 2. Run Exploratory Data Analysis (EDA)
```bash
python scripts/01_eda.py
```
Outputs statistical distributions, missing-value audits, and sample trajectory maps to `results/eda/` and `maps/`.

### 3. Build Feature Tables
```bash
python scripts/02_build_dataset.py
```
Extracts all 26 leak-free features across 80 training trips and 10 student-test trips into `data_cache/train_features.parquet` and `data_cache/student_test_features.parquet`.

### 4. Train & Compare Models
```bash
python scripts/03_train_compare.py
```
Performs 5-Fold GroupKFold CV across 6 model families, evaluates on student test trips, and serializes trained models to `artifacts/`.

### 5. Generate Diagnostic Plots & Folium Maps
```bash
python scripts/04_plots.py
python scripts/06_folium_maps.py
```

---

## 🗺️ Interactive Folium Visualization

Interactive HTML maps with layer controls are located in `maps/`:
- **Best Trip (00000052):** `maps/student_test_best_00000052.html` (95.04% Top-1 Accuracy)
- **Typical Trip (00000056):** `maps/student_test_typical_00000056.html` (87.11% Top-1 Accuracy)
- **Worst Trip (00000077):** `maps/student_test_worst_00000077.html` (76.99% Top-1 Accuracy)

---

## 👥 Authors
- **Team PathSnap** | PES University (UE24CS352A)
