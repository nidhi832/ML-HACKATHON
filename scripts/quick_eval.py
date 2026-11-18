import os, sys, time
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
sys.path.insert(0, os.getcwd())
import joblib, pandas as pd
from src.models import FEATURE_COLS
from src.evaluate import compute_candidate_metrics, compute_observation_top1

df = pd.read_parquet("data_cache/student_test_features.parquet")
X = df[FEATURE_COLS].to_numpy()
rows = []
for name in ["logistic_regression", "gaussian_nb", "subsampled_knn", "random_forest", "lightgbm"]:
    m = joblib.load(f"artifacts/{name}.joblib")
    t = time.time(); p = m.predict_proba(X); dt = time.time() - t
    met = compute_candidate_metrics(df["y"].to_numpy(), p)
    d = df[["trip_id", "obs_idx", "arc_id", "y"]].copy(); d["prob"] = p
    rows.append(dict(model=name, top1_pct=round(compute_observation_top1(d) * 100, 2),
                     macro_f1=round(met["candidate_macro_f1"], 4),
                     log_loss=round(met["log_loss"], 4), infer_s=round(dt, 1)))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_csv("results/quick_student_eval.csv", index=False)
print(out.to_string(index=False))