# TRACE.AI & PathSnap: Multi-Laptop Deployment Guide

This guide describes the complete, step-by-step instructions for deploying and running the **PathSnap / TRACE.AI** map-matching system across two separate physical machines (e.g., during live competition judging or distributed fleet operations):

- **Laptop 1 (Inference Server):** Hosts the trained machine learning models, spatial indexing engine, HMM Viterbi sequence decoder, and FastAPI REST API on port `8000`.
- **Laptop 2 (Client / Evaluator / Web UI):** Sends raw GPS trips, invokes evaluation scripts, queries REST endpoints, and views the geospatial dashboard.

---

## 1. Network Prerequisites
Both laptops must be connected to the **same local area network (LAN)** or Wi-Fi hotspot.

### Step 1.1: Determine Laptop 1's LAN IP Address
On **Laptop 1**, open PowerShell or Command Prompt and run:
```powershell
ipconfig
```
Look for **IPv4 Address** under your active Wi-Fi or Ethernet adapter (e.g., `192.168.1.105` or `10.0.0.42`).  
Throughout this guide, replace `<LAPTOP_1_IP>` with this address.

---

## 2. Laptop 1: Server Configuration & Startup

### Step 2.1: Open Windows Firewall for Inbound Port 8000
Run PowerShell as **Administrator** on Laptop 1:
```powershell
New-NetFirewallRule -DisplayName "TRACE_AI_API_8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```
*(Alternatively, in Windows Defender Firewall with Advanced Security -> Inbound Rules -> New Rule -> Port -> TCP 8000 -> Allow the connection).*

### Step 2.2: Configure Evaluator Trip Directory (Optional)
By default, the server evaluates trips in `Student folder/student_test`. If competition judges provide an unseen evaluation directory:
```powershell
# Set EVAL_DATA_DIR environment variable
$env:EVAL_DATA_DIR = "C:\Path\To\Hidden_Evaluator_Trips"
```
Or in Command Prompt:
```cmd
set EVAL_DATA_DIR=C:\Path\To\Hidden_Evaluator_Trips
```

### Step 2.3: Start the Backend Service
Start FastAPI listening on `0.0.0.0` (all network interfaces):
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Or simply run:
```cmd
launch_trace_ai.bat
```

You should see:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 3. Laptop 2: Client Verification & Evaluation

### Step 3.1: Verify Connectivity via cURL
From **Laptop 2** terminal or PowerShell:
```powershell
curl http://<LAPTOP_1_IP>:8000/health
```
Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "LightGBM",
  "hmm_decoder_loaded": true,
  "student_test_trips_count": 10,
  "system": "TRACE.AI Core Engine",
  "version": "1.0.0"
}
```

### Step 3.2: Run the Automated Pre-Evaluation Checklist
Laptop 2 can run the built-in end-to-end verification script against Laptop 1:
```powershell
python scripts/pre_evaluation_check.py --url http://<LAPTOP_1_IP>:8000
```
This tests:
1. `GET /health` (Model & HMM status)
2. `GET /trips` (Discovered trip enumeration)
3. `POST /predict` (Live prediction on all 10 trips)
4. Decoder toggle (`use_hmm=False` -> Classifier Argmax vs `use_hmm=True` -> HMM Viterbi)
5. Prints a formatted PASS/FAIL summary table with latency and continuity metrics.

### Step 3.3: Access the TRACE.AI Geospatial Web Dashboard
You have two options to view the interactive UI on Laptop 2:

#### Option A: Direct Web Browser Access (Zero-Setup)
Since the production frontend is pre-built into `api/main.py`, open your browser on Laptop 2 and navigate to:
```text
http://<LAPTOP_1_IP>:8000
```
The full TRACE.AI interactive geospatial dashboard will load immediately!

#### Option B: Dedicated Frontend Dev Server
If running the frontend independently on Laptop 2:
```powershell
cd ui
$env:VITE_API_BASE_URL = "http://<LAPTOP_1_IP>:8000"
npm run dev -- --host
```
Open `http://localhost:5173`. Alternatively, navigate to **Settings** in the web UI, enter `http://<LAPTOP_1_IP>:8000` in the **Backend API Server Base URL** field, click **Test Ping**, and click **Save Settings**.

---

## 4. Evaluator API Specification

### Endpoint 1: Health Check
- **Method:** `GET`
- **URL:** `http://<LAPTOP_1_IP>:8000/health`
- **Response:** JSON confirming model, HMM, and trip count.

### Endpoint 2: List Available Trips
- **Method:** `GET`
- **URL:** `http://<LAPTOP_1_IP>:8000/trips?partition=student_test`
- **Response:** List of trip directory IDs found in `EVAL_DATA_DIR`.

### Endpoint 3: Predict Trip Route
- **Method:** `POST`
- **URL:** `http://<LAPTOP_1_IP>:8000/predict`
- **Headers:** `Content-Type: application/json`
- **Body:**
  ```json
  {
    "trip_id": "00000052",
    "partition": "student_test",
    "model_name": "lightgbm",
    "use_hmm": true
  }
  ```
- **Response:**
  ```json
  {
    "trip_id": "00000052",
    "partition": "student_test",
    "num_observations": 262,
    "predicted_arcs": [1284, 1284, 1285, ...],
    "traversed_route": [1284, 1285, 1289, ...],
    "num_arcs_traversed": 85,
    "continuity_pct": 96.93,
    "model_name": "LightGBM",
    "decoder": "HMM Viterbi",
    "status": "success",
    "latency_ms": 64.3,
    "top1_accuracy_pct": 97.71,
    "route_lcs_f1": 0.9444
  }
  ```
  *(Note: `top1_accuracy_pct` and `route_lcs_f1` are returned strictly when `.route` ground truth exists in the trip folder; on blind test trips, they return `null` while `continuity_pct` is computed purely from network graph geometry).*

---

## 5. Troubleshooting & Diagnostics

| Symptom | Cause | Solution |
|---|---|---|
| `curl: (7) Failed to connect to <IP> port 8000` | Windows Firewall blocking inbound connections on Laptop 1 | Run the firewall command in Step 2.1 on Laptop 1. Verify both machines are on the same subnet. |
| `Trip '000000xx' not found` | `EVAL_DATA_DIR` does not point to the evaluation folder | Ensure `$env:EVAL_DATA_DIR` points to the directory containing trip folders (each containing `.track`, `.nodes`, `.arcs`). |
| CORS Error in browser console | Origin mismatch | `api/main.py` has `CORSMiddleware` configured with `allow_origins=["*"]`. Ensure requests go to port 8000. |
| Port 8000 in use | Another process is using port 8000 | Kill process: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force` |
