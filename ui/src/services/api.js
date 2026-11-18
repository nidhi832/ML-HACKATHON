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

export const FALLBACK_HMM_ABLATION = [
  {
    config_name: "Classifier-Only Baseline (LightGBM Argmax, no HMM)",
    transition_weight: 0.0,
    reverse_arc_log_prob: 0.0,
    two_hop_log_prob: 0.0,
    three_hop_log_prob: "",
    gap_threshold_sec: "",
    top1_acc_pct: 87.16,
    route_continuity_pct: 93.93,
    forward_continuity_pct: 89.86,
    reverse_uturns: 779,
    disconnected_jumps: 685,
    route_lcs_f1: 0.7511,
    route_jaccard: 0.797,
    decode_ms_per_1000_obs: 0.0
  },
  {
    config_name: "Basic Topology HMM (no U-turn penalty, tr_w=1.5)",
    transition_weight: 1.5,
    reverse_arc_log_prob: -0.5,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: 30.0,
    top1_acc_pct: 90.91,
    route_continuity_pct: 99.12,
    forward_continuity_pct: 97.02,
    reverse_uturns: 402,
    disconnected_jumps: 24,
    route_lcs_f1: 0.8916,
    route_jaccard: 0.9002,
    decode_ms_per_1000_obs: 94.64
  },
  {
    config_name: "Reverse-Arc-Aware HMM (rev=-10, tr_w=1.0)",
    transition_weight: 1.0,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: 30.0,
    top1_acc_pct: 92.27,
    route_continuity_pct: 98.51,
    forward_continuity_pct: 98.49,
    reverse_uturns: 5,
    disconnected_jumps: 24,
    route_lcs_f1: 0.938,
    route_jaccard: 0.9055,
    decode_ms_per_1000_obs: 93.86
  },
  {
    config_name: "Reverse-Arc-Aware HMM (rev=-10, tr_w=1.5)",
    transition_weight: 1.5,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: 30.0,
    top1_acc_pct: 92.91,
    route_continuity_pct: 98.79,
    forward_continuity_pct: 98.77,
    reverse_uturns: 3,
    disconnected_jumps: 24,
    route_lcs_f1: 0.9474,
    route_jaccard: 0.9161,
    decode_ms_per_1000_obs: 92.81
  },
  {
    config_name: "Champion HMM (rev=-10, tr_w=2.0, gap=30s) [Selected]",
    transition_weight: 2.0,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: 30.0,
    top1_acc_pct: 92.87,
    route_continuity_pct: 98.96,
    forward_continuity_pct: 98.95,
    reverse_uturns: 3,
    disconnected_jumps: 23,
    route_lcs_f1: 0.9502,
    route_jaccard: 0.9146,
    decode_ms_per_1000_obs: 89.37
  },
  {
    config_name: "Reverse-Arc-Aware HMM (rev=-10, tr_w=2.5)",
    transition_weight: 2.5,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: 30.0,
    top1_acc_pct: 93.0,
    route_continuity_pct: 99.08,
    forward_continuity_pct: 99.07,
    reverse_uturns: 3,
    disconnected_jumps: 23,
    route_lcs_f1: 0.9544,
    route_jaccard: 0.9189,
    decode_ms_per_1000_obs: 97.82
  },
  {
    config_name: "Reverse-Arc-Aware HMM + 3-Hop (rev=-10, h3=-5, tr_w=2.0)",
    transition_weight: 2.0,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: -5.0,
    gap_threshold_sec: 30.0,
    top1_acc_pct: 92.82,
    route_continuity_pct: 99.0,
    forward_continuity_pct: 98.98,
    reverse_uturns: 3,
    disconnected_jumps: 34,
    route_lcs_f1: 0.9501,
    route_jaccard: 0.9138,
    decode_ms_per_1000_obs: 102.77
  },
  {
    config_name: "No GPS Gap Softening (rev=-10, tr_w=2.0, gap=disabled)",
    transition_weight: 2.0,
    reverse_arc_log_prob: -10.0,
    two_hop_log_prob: -2.5,
    three_hop_log_prob: "",
    gap_threshold_sec: "",
    top1_acc_pct: 92.87,
    route_continuity_pct: 98.96,
    forward_continuity_pct: 98.95,
    reverse_uturns: 3,
    disconnected_jumps: 23,
    route_lcs_f1: 0.9502,
    route_jaccard: 0.9146,
    decode_ms_per_1000_obs: 96.99
  }
];

export const FALLBACK_HMM_COMPARISON = [
  {
    model_key: "lightgbm",
    model_name: "LightGBM",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 87.16,
    hmm_top1_acc_pct: 92.87,
    delta_top1_acc_pct: 5.71,
    clf_route_continuity_pct: 93.93,
    hmm_route_continuity_pct: 98.96,
    delta_route_continuity_pct: 5.03,
    clf_forward_continuity_pct: 89.86,
    hmm_forward_continuity_pct: 98.95,
    delta_forward_continuity_pct: 9.08,
    clf_reverse_uturns: 779,
    hmm_reverse_uturns: 3,
    clf_disconnected_jumps: 685,
    hmm_disconnected_jumps: 23,
    invalid_jumps_removed: 662,
    clf_route_lcs_f1: 0.7511,
    hmm_route_lcs_f1: 0.9502,
    delta_route_lcs_f1: 0.1991,
    clf_route_jaccard: 0.797,
    hmm_route_jaccard: 0.9146,
    errors_fixed_by_hmm: 1366,
    errors_introduced_by_hmm: 272,
    net_obs_fixed: 1094,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 5590.1,
    hmm_decode_ms_per_1000_obs: 291.76
  },
  {
    model_key: "random_forest",
    model_name: "Random Forest",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 85.83,
    hmm_top1_acc_pct: 92.47,
    delta_top1_acc_pct: 6.64,
    clf_route_continuity_pct: 93.2,
    hmm_route_continuity_pct: 98.96,
    delta_route_continuity_pct: 5.76,
    clf_forward_continuity_pct: 88.14,
    hmm_forward_continuity_pct: 98.95,
    delta_forward_continuity_pct: 10.8,
    clf_reverse_uturns: 968,
    hmm_reverse_uturns: 2,
    clf_disconnected_jumps: 789,
    hmm_disconnected_jumps: 23,
    invalid_jumps_removed: 766,
    clf_route_lcs_f1: 0.7297,
    hmm_route_lcs_f1: 0.9494,
    delta_route_lcs_f1: 0.2197,
    clf_route_jaccard: 0.781,
    hmm_route_jaccard: 0.9159,
    errors_fixed_by_hmm: 1564,
    errors_introduced_by_hmm: 291,
    net_obs_fixed: 1273,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 2832.25,
    hmm_decode_ms_per_1000_obs: 147.82
  },
  {
    model_key: "pytorch_mlp",
    model_name: "MLP (PyTorch)",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 85.86,
    hmm_top1_acc_pct: 91.12,
    delta_top1_acc_pct: 5.27,
    clf_route_continuity_pct: 92.96,
    hmm_route_continuity_pct: 98.95,
    delta_route_continuity_pct: 5.99,
    clf_forward_continuity_pct: 88.99,
    hmm_forward_continuity_pct: 98.93,
    delta_forward_continuity_pct: 9.94,
    clf_reverse_uturns: 760,
    hmm_reverse_uturns: 3,
    clf_disconnected_jumps: 794,
    hmm_disconnected_jumps: 23,
    invalid_jumps_removed: 771,
    clf_route_lcs_f1: 0.7343,
    hmm_route_lcs_f1: 0.9514,
    delta_route_lcs_f1: 0.2171,
    clf_route_jaccard: 0.7763,
    hmm_route_jaccard: 0.919,
    errors_fixed_by_hmm: 1328,
    errors_introduced_by_hmm: 319,
    net_obs_fixed: 1009,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 2269.36,
    hmm_decode_ms_per_1000_obs: 118.44
  },
  {
    model_key: "logistic_regression",
    model_name: "Logistic Regression",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 84.67,
    hmm_top1_acc_pct: 90.61,
    delta_top1_acc_pct: 5.94,
    clf_route_continuity_pct: 92.03,
    hmm_route_continuity_pct: 99.12,
    delta_route_continuity_pct: 7.09,
    clf_forward_continuity_pct: 86.99,
    hmm_forward_continuity_pct: 99.11,
    delta_forward_continuity_pct: 12.13,
    clf_reverse_uturns: 966,
    hmm_reverse_uturns: 1,
    clf_disconnected_jumps: 875,
    hmm_disconnected_jumps: 22,
    invalid_jumps_removed: 853,
    clf_route_lcs_f1: 0.71,
    hmm_route_lcs_f1: 0.9596,
    delta_route_lcs_f1: 0.2496,
    clf_route_jaccard: 0.7647,
    hmm_route_jaccard: 0.9316,
    errors_fixed_by_hmm: 1580,
    errors_introduced_by_hmm: 442,
    net_obs_fixed: 1138,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 2616.28,
    hmm_decode_ms_per_1000_obs: 136.55
  },
  {
    model_key: "subsampled_knn",
    model_name: "K-Nearest Neighbours",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 80.64,
    hmm_top1_acc_pct: 85.42,
    delta_top1_acc_pct: 4.79,
    clf_route_continuity_pct: 87.91,
    hmm_route_continuity_pct: 97.46,
    delta_route_continuity_pct: 9.55,
    clf_forward_continuity_pct: 84.66,
    hmm_forward_continuity_pct: 97.4,
    delta_forward_continuity_pct: 12.74,
    clf_reverse_uturns: 622,
    hmm_reverse_uturns: 12,
    clf_disconnected_jumps: 1500,
    hmm_disconnected_jumps: 23,
    invalid_jumps_removed: 1477,
    clf_route_lcs_f1: 0.6671,
    hmm_route_lcs_f1: 0.8983,
    delta_route_lcs_f1: 0.2312,
    clf_route_jaccard: 0.7252,
    hmm_route_jaccard: 0.8849,
    errors_fixed_by_hmm: 1451,
    errors_introduced_by_hmm: 534,
    net_obs_fixed: 917,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 1811.1,
    hmm_decode_ms_per_1000_obs: 94.52
  },
  {
    model_key: "gaussian_nb",
    model_name: "Gaussian Naive Bayes",
    num_trips: 10,
    total_observations: 19160,
    candidate_recall_pct: 98.83,
    clf_top1_acc_pct: 81.27,
    hmm_top1_acc_pct: 89.87,
    delta_top1_acc_pct: 8.6,
    clf_route_continuity_pct: 91.19,
    hmm_route_continuity_pct: 99.42,
    delta_route_continuity_pct: 8.23,
    clf_forward_continuity_pct: 84.34,
    hmm_forward_continuity_pct: 99.42,
    delta_forward_continuity_pct: 15.08,
    clf_reverse_uturns: 1312,
    hmm_reverse_uturns: 0,
    clf_disconnected_jumps: 1018,
    hmm_disconnected_jumps: 22,
    invalid_jumps_removed: 996,
    clf_route_lcs_f1: 0.6662,
    hmm_route_lcs_f1: 0.9628,
    delta_route_lcs_f1: 0.2966,
    clf_route_jaccard: 0.7501,
    hmm_route_jaccard: 0.9358,
    errors_fixed_by_hmm: 2284,
    errors_introduced_by_hmm: 636,
    net_obs_fixed: 1648,
    gap_softened_steps: 2,
    hmm_decode_total_ms: 2330.96,
    hmm_decode_ms_per_1000_obs: 121.66
  }
];

export const FALLBACK_TUNING_COMPARISON = [
  {
    model: "LightGBM Baseline",
    cv_macro_f1_mean: 0.913,
    cv_macro_f1_std: 0.0192,
    student_top1_acc: 87.16,
    student_macro_f1: 0.9293,
    log_loss: 0.0613,
    train_time_s: 28.45,
    infer_time_ms_per_1000_obs: 3.67
  },
  {
    model: "LightGBM Bayesian Tuned",
    cv_macro_f1_mean: 0.917,
    cv_macro_f1_std: 0.0157,
    student_top1_acc: 86.89,
    student_macro_f1: 0.9275,
    log_loss: 0.0624,
    train_time_s: 34.12,
    infer_time_ms_per_1000_obs: 4.02
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
    hmm_comparison: FALLBACK_HMM_COMPARISON,
    hmm_ablation: FALLBACK_HMM_ABLATION,
    hmm_per_trip: [],
    tuning_comparison: FALLBACK_TUNING_COMPARISON,
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
