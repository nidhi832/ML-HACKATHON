/**
 * TRACE.AI API Client
 * Configurable API Client with support for multi-laptop deployment (Laptop 1 API <-> Laptop 2 UI).
 * Supports custom API base URL via localStorage ('TRACE_AI_API_URL') or VITE_API_BASE_URL.
 */

export function getApiBaseUrl() {
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem('TRACE_AI_API_URL');
    if (custom && custom.trim()) {
      return custom.trim().replace(/\/$/, '');
    }
  }
  const envUrl = import.meta.env?.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/$/, '');
  }
  return '';
}

export function setApiBaseUrl(url) {
  if (typeof window !== 'undefined') {
    if (!url || !url.trim()) {
      localStorage.removeItem('TRACE_AI_API_URL');
    } else {
      localStorage.setItem('TRACE_AI_API_URL', url.trim().replace(/\/$/, ''));
    }
  }
}

// Fallback real results from PathSnap experiment runs
export const FALLBACK_MODELS = [
  {
    model_name: "LightGBM",
    algorithm_family: "Ensemble (Gradient Boosting)",
    cv_macro_f1_mean: 0.9008,
    cv_macro_f1_std: 0.0435,
    cv_train_macro_f1: 0.9517,
    train_val_gap: 0.0510,
    student_top1_acc: 87.16,
    route_continuity_pct: 98.96,
    student_macro_f1: 0.9293,
    log_loss: 0.0613,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 45.01,
    is_champion: true
  },
  {
    model_name: "Random Forest",
    algorithm_family: "Ensemble (Bagging)",
    cv_macro_f1_mean: 0.9042,
    cv_macro_f1_std: 0.0360,
    cv_train_macro_f1: 0.9379,
    train_val_gap: 0.0337,
    student_top1_acc: 85.83,
    route_continuity_pct: 98.96,
    student_macro_f1: 0.9237,
    log_loss: 0.0659,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 25.49,
    is_champion: false
  },
  {
    model_name: "MLP (PyTorch Neural Network)",
    algorithm_family: "Deep Learning",
    cv_macro_f1_mean: 0.8999,
    cv_macro_f1_std: 0.0369,
    cv_train_macro_f1: 0.9274,
    train_val_gap: 0.0275,
    student_top1_acc: 85.86,
    route_continuity_pct: 98.95,
    student_macro_f1: 0.9224,
    log_loss: 0.0700,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 126.73,
    is_champion: false
  },
  {
    model_name: "Logistic Regression",
    algorithm_family: "Generalized Linear",
    cv_macro_f1_mean: 0.8743,
    cv_macro_f1_std: 0.0301,
    cv_train_macro_f1: 0.8824,
    train_val_gap: 0.0081,
    student_top1_acc: 84.67,
    route_continuity_pct: 99.12,
    student_macro_f1: 0.8989,
    log_loss: 0.0991,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 1.42,
    is_champion: false
  },
  {
    model_name: "Gaussian Naive Bayes",
    algorithm_family: "Probabilistic",
    cv_macro_f1_mean: 0.6671,
    cv_macro_f1_std: 0.0311,
    cv_train_macro_f1: 0.6558,
    train_val_gap: -0.0113,
    student_top1_acc: 81.27,
    route_continuity_pct: 99.42,
    student_macro_f1: 0.6583,
    log_loss: 1.4502,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 4.08,
    is_champion: false
  },
  {
    model_name: "K-Nearest Neighbours",
    algorithm_family: "Instance-based",
    cv_macro_f1_mean: 0.8551,
    cv_macro_f1_std: 0.0400,
    cv_train_macro_f1: 0.8890,
    train_val_gap: 0.0339,
    student_top1_acc: 80.64,
    route_continuity_pct: 97.46,
    student_macro_f1: 0.8742,
    log_loss: 0.1684,
    candidate_miss_rate_pct: 1.17,
    infer_time_ms_per_1000_obs: 524.19,
    is_champion: false
  }
];

export async function fetchHealth() {
  const base = getApiBaseUrl();
  const url = `${base}/health`;
  const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
  if (!res.ok) throw new Error(`Health check failed with status ${res.status}`);
  return await res.json();
}

export async function fetchOverview() {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/api/overview`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend /api/overview unavailable, using verified offline cache");
  }

  return {
    kpis: {
      total_trips: 90,
      total_gps_points: 227577,
      top1_accuracy_pct: 92.87,
      candidate_miss_rate_pct: 1.17,
      candidate_recall_pct: 98.83,
      avg_processing_time_ms: 45.01,
      route_continuity_pct: 98.96,
      cv_macro_f1: 0.9008,
      student_macro_f1: 0.9293
    },
    featured_trip_id: '00000052',
    model_champion: 'LightGBM + HMM Viterbi Decoder'
  };
}

export async function fetchTripsList(partition = 'student_test') {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/api/trips?partition=${partition}`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend /api/trips unavailable, using verified offline trip list");
  }

  // Real verified student_test trips
  return [
    { trip_id: "00000000", partition: "student_test", gps_points: 2503, route_segments: 179, accuracy_pct: 94.93, status: "Ready", distance_km: 37.55, duration_min: 41.7 },
    { trip_id: "00000007", partition: "student_test", gps_points: 1354, route_segments: 123, accuracy_pct: 99.26, status: "Ready", distance_km: 20.31, duration_min: 22.6 },
    { trip_id: "00000009", partition: "student_test", gps_points: 1049, route_segments: 429, accuracy_pct: 87.99, status: "Ready", distance_km: 15.74, duration_min: 17.5 },
    { trip_id: "00000040", partition: "student_test", gps_points: 2995, route_segments: 309, accuracy_pct: 94.39, status: "Ready", distance_km: 44.93, duration_min: 49.9 },
    { trip_id: "00000052", partition: "student_test", gps_points: 262, route_segments: 95, accuracy_pct: 97.71, status: "Ready", distance_km: 3.93, duration_min: 4.4 },
    { trip_id: "00000056", partition: "student_test", gps_points: 3958, route_segments: 1323, accuracy_pct: 93.66, status: "Ready", distance_km: 59.37, duration_min: 66.0 },
    { trip_id: "00000065", partition: "student_test", gps_points: 768, route_segments: 145, accuracy_pct: 88.80, status: "Ready", distance_km: 11.52, duration_min: 12.8 },
    { trip_id: "00000076", partition: "student_test", gps_points: 1486, route_segments: 374, accuracy_pct: 94.28, status: "Ready", distance_km: 22.29, duration_min: 24.8 },
    { trip_id: "00000077", partition: "student_test", gps_points: 3911, route_segments: 268, accuracy_pct: 88.49, status: "Ready", distance_km: 58.67, duration_min: 65.2 },
    { trip_id: "00000091", partition: "student_test", gps_points: 874, route_segments: 232, accuracy_pct: 93.36, status: "Ready", distance_km: 13.11, duration_min: 14.6 },
  ];
}
export const fetchTrips = fetchTripsList;

export async function fetchTripDetails(tripId, partition = 'student_test', modelName = 'lightgbm', useHmm = true) {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/api/trips/${tripId}?partition=${partition}&model_name=${modelName}&use_hmm=${useHmm}`, {
      signal: AbortSignal.timeout(6000)
    });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn(`Backend trip details unavailable for ${tripId}`);
    throw err;
  }
}

export async function runMapMatching(req) {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: AbortSignal.timeout(15000)
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Match API failed (${res.status}): ${errText}`);
  }
  return await res.json();
}

export async function fetchModelsComparison() {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/api/models`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend /api/models unavailable, using fallback data");
  }

  return {
    models: FALLBACK_MODELS,
    confusion_matrices: {
      lightgbm: { tp: 17290, fp: 1870, fn: 226, tn: 172214 }
    },
    mlp_epoch_loss: [0.2683, 0.1115, 0.0894, 0.0801, 0.0752, 0.0718, 0.0694, 0.0678, 0.0667, 0.0661],
    hmm_comparison: [],
    hmm_ablation: [],
    hmm_per_trip: [],
    tuning_comparison: [],
    optuna_trials: [],
    group_kfold_summary: {
      folds: 5,
      grouping_key: "trip_id",
      train_samples: 2084170,
      test_samples: 191600
    }
  };
}

export async function fetchEvaluationDashboard() {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/api/evaluation`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend /api/evaluation unavailable");
    throw err;
  }
}
export const fetchEvaluation = fetchEvaluationDashboard;
