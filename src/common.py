"""Shared paths, constants and helpers used by the clustering/segment scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42
N_INIT = 20


def load_rfm():
    """Load raw and scaled customer RFM tables and validate they are aligned."""
    scaled_path = RESULTS / "rfm_scaled.csv"
    raw_path = RESULTS / "customer_rfm.csv"
    if not scaled_path.exists() or not raw_path.exists():
        raise FileNotFoundError("Run python src/build_rfm.py before this script.")
    scaled = pd.read_csv(scaled_path)
    raw = pd.read_csv(raw_path)
    required = {"Customer ID", *FEATURES}
    if not required.issubset(scaled.columns) or not required.issubset(raw.columns):
        raise ValueError("RFM input files are missing required columns.")
    if len(scaled) != len(raw) or not scaled["Customer ID"].equals(raw["Customer ID"]):
        raise ValueError("Raw and scaled RFM customer rows/order do not match.")
    if scaled["Customer ID"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")
    X = scaled[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Scaled RFM contains NaN or infinite values.")
    return raw, scaled, X


def fit_kmeans(X, k, n_init=N_INIT, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init,
                  algorithm="lloyd").fit(X)


def kmeans_stability(X, k, n_boot=20, frac=0.8, seed=RANDOM_STATE, n_init=10):
    """Mean/std Adjusted Rand Index between the full-data K-Means labelling and
    labellings predicted by models trained on random 80% subsamples."""
    rng = np.random.default_rng(seed)
    reference = fit_kmeans(X, k, n_init=N_INIT, seed=seed).labels_
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        model = KMeans(n_clusters=k, random_state=seed + b + 1, n_init=n_init).fit(X[idx])
        scores.append(adjusted_rand_score(reference, model.predict(X)))
    return float(np.mean(scores)), float(np.std(scores))
