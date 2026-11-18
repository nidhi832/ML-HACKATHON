"""
src/models.py - Model definitions across 6 diverse algorithm families:
1. Logistic Regression (Linear)
2. Gaussian Naive Bayes (Probabilistic / Bayesian)
3. K-Nearest Neighbours (Instance-based)
4. Random Forest (Ensemble / Bagging)
5. LightGBM (Gradient Boosting)
6. MLP (Deep Learning / Neural Network with epoch loss history)
"""

import os
import sys

# Prevent joblib loky from calling external subprocess to count CPUs in Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: 4
except Exception:
    pass

from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLS = [
    "dist_m", "proj_frac", "raw_proj_frac", "arc_length_m", "candidate_rank",
    "dist_gap_to_best_m", "n_cand_within_10m", "n_cand_within_25m", "n_cand_within_50m",
    "arc_bearing", "gps_heading", "heading_abs_diff", "heading_cos_diff",
    "speed_mps", "is_stationary", "turn_angle_deg", "acceleration_mps2",
    "dt_prev", "dt_next", "prev_point_dist_to_arc", "next_point_dist_to_arc",
    "rolling_dist_mean_3", "dist_change_prev", "arc_out_degree", "arc_in_degree",
    "has_reverse_arc",
]


class BaseCandidateModel:
    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError


class LogisticRegressionModel(BaseCandidateModel):
    def __init__(self, C: float = 1.0, max_iter: int = 250, random_state: int = 42):
        super().__init__(name="Logistic Regression", family="Linear / Generalized Linear")
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs"
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": "lbfgs",
            "scaler": "StandardScaler",
            "random_state": self.random_state,
        }


class GaussianNBModel(BaseCandidateModel):
    def __init__(self, var_smoothing: float = 1e-8):
        super().__init__(name="Gaussian Naive Bayes", family="Bayesian / Probabilistic")
        self.var_smoothing = var_smoothing
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=var_smoothing)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "var_smoothing": self.var_smoothing,
            "scaler": "StandardScaler",
        }


class SubsampledKNNModel(BaseCandidateModel):
    def __init__(self, n_neighbors: int = 9, max_fit_samples: int = 25000, random_state: int = 42):
        super().__init__(name="K-Nearest Neighbours", family="Instance-based")
        self.n_neighbors = n_neighbors
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", algorithm="brute", n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        N = len(X)
        if N > self.max_fit_samples:
            # Stratified representative subsample
            np.random.seed(self.random_state)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            pos_sample_sz = int(self.max_fit_samples * (len(pos_idx) / N))
            neg_sample_sz = self.max_fit_samples - pos_sample_sz
            sub_pos = np.random.choice(pos_idx, size=min(pos_sample_sz, len(pos_idx)), replace=False)
            sub_neg = np.random.choice(neg_idx, size=min(neg_sample_sz, len(neg_idx)), replace=False)
            sub_idx = np.concatenate([sub_pos, sub_neg])
            np.random.shuffle(sub_idx)
            X_fit = X_scaled[sub_idx]
            y_fit = y[sub_idx]
        else:
            X_fit = X_scaled
            y_fit = y

        self.clf.fit(X_fit, y_fit)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # Predict in chunks to conserve memory
        chunk_sz = 10000
        probs = []
        for i in range(0, len(X_scaled), chunk_sz):
            p = self.clf.predict_proba(X_scaled[i:i + chunk_sz])[:, 1]
            probs.append(p)
        return np.concatenate(probs).astype(np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "n_neighbors": self.n_neighbors,
            "weights": "distance",
            "max_fit_samples": self.max_fit_samples,
            "random_state": self.random_state,
        }


class RandomForestModel(BaseCandidateModel):
    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 12,
        max_samples: float = 0.10,
        random_state: int = 42,
    ):
        super().__init__(name="Random Forest", family="Ensemble (Bagging)")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_samples": self.max_samples,
            "min_samples_split": 10,
            "random_state": self.random_state,
        }


class LightGBMModel(BaseCandidateModel):
    def __init__(
        self,
        n_estimators: int = 180,
        max_depth: int = 7,
        num_leaves: int = 63,
        learning_rate: float = 0.08,
        random_state: int = 42,
    ):
        super().__init__(name="LightGBM", family="Ensemble (Gradient Boosting)")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }


class PyTorchMLPNet(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPModel(BaseCandidateModel):
    def __init__(
        self,
        in_features: int = len(FEATURE_COLS),
        epochs: int = 10,
        batch_size: int = 4096,
        lr: float = 0.005,
        random_state: int = 42,
    ):
        super().__init__(name="MLP (PyTorch Neural Network)", family="Deep Learning / Neural Networks")
        self.in_features = in_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.net = PyTorchMLPNet(in_features)
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Subsample to a balanced representative set for fast CPU epoch convergence if huge
        N = len(X_scaled)
        max_train = 250000
        if N > max_train:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            pos_sz = int(max_train * (len(pos_idx) / N))
            neg_sz = max_train - pos_sz
            sub_pos = np.random.choice(pos_idx, size=min(pos_sz, len(pos_idx)), replace=False)
            sub_neg = np.random.choice(neg_idx, size=min(neg_sz, len(neg_idx)), replace=False)
            sub_idx = np.concatenate([sub_pos, sub_neg])
            np.random.shuffle(sub_idx)
            X_tensor = torch.tensor(X_scaled[sub_idx], dtype=torch.float32)
            y_tensor = torch.tensor(y[sub_idx], dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.net.train()
        self.loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            epoch_loss = float(total_loss / len(dataset))
            self.loss_history.append(epoch_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        probs = []
        with torch.no_grad():
            for (bx,) in loader:
                logits = self.net(bx)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs).astype(np.float64)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "family": self.family,
            "architecture": "Linear(64)->BN->ReLU->Dropout(0.1)->Linear(32)->BN->ReLU->Linear(1)",
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "scaler": "StandardScaler",
            "random_state": self.random_state,
        }
