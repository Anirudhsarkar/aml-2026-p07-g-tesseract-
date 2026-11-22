"""Phase 4 K-Means baseline: evaluate k=2..10 on scaled customer RFM.

Run from repository root: python src/clustering.py
Silhouette uses a deterministic 2,000-row sample for runtime. Other metrics
use all customers. The best silhouette is only a candidate, not final k.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FEATURES = ["Recency", "Frequency", "Monetary"]
K_VALUES = range(2, 11)
RANDOM_STATE = 42
N_INIT = 20
SILHOUETTE_SAMPLE_SIZE = 2000


def load_data():
    scaled_path = RESULTS / "rfm_scaled.csv"
    raw_path = RESULTS / "customer_rfm.csv"
    if not scaled_path.exists() or not raw_path.exists():
        raise FileNotFoundError("Run python src/build_rfm.py before clustering.")
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


def main():
    raw, scaled, X = load_data()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X))
    metric_rows, profile_rows = [], []
    assignments = pd.DataFrame({"Customer ID": scaled["Customer ID"]})

    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                       n_init=N_INIT, algorithm="lloyd")
        labels = model.fit_predict(X)
        metric_rows.append({
            "k": k, "inertia": float(model.inertia_),
            "silhouette_sampled": float(silhouette_score(
                X, labels, sample_size=sample_size, random_state=RANDOM_STATE)),
            "silhouette_sample_size": sample_size,
            "silhouette_full": float(silhouette_score(X, labels)),
            "min_cluster_share": float(np.bincount(labels).min() / len(labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "random_state": RANDOM_STATE, "n_init": N_INIT
        })
        assignments[f"Cluster_k{k}"] = labels
        profile = raw[["Customer ID", *FEATURES]].copy()
        profile["Cluster"] = labels
        grouped = profile.groupby("Cluster").agg(
            customers=("Customer ID", "count"),
            recency_mean=("Recency", "mean"), recency_median=("Recency", "median"),
            frequency_mean=("Frequency", "mean"), frequency_median=("Frequency", "median"),
            monetary_mean=("Monetary", "mean"), monetary_median=("Monetary", "median")
        ).reset_index()
        profile_rows.extend({"k": k, **row} for row in grouped.to_dict("records"))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "kmeans_metrics.csv", index=False)
    assignments.to_csv(RESULTS / "kmeans_cluster_assignments.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(RESULTS / "kmeans_cluster_profiles.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics.k, metrics.inertia, marker="o")
    plt.xticks(list(K_VALUES)); plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia"); plt.title("K-Means Elbow Curve"); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIGURES / "kmeans_elbow_curve.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics.k, metrics.silhouette_sampled, marker="o")
    plt.xticks(list(K_VALUES)); plt.xlabel("Number of clusters (k)")
    plt.ylabel(f"Silhouette score (sample n={sample_size})")
    plt.title("K-Means Silhouette Scores"); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIGURES / "kmeans_silhouette_scores.png", dpi=160); plt.close()

    provisional_k = int(metrics.loc[metrics.silhouette_sampled.idxmax(), "k"])
    labels = KMeans(n_clusters=provisional_k, random_state=RANDOM_STATE,
                    n_init=N_INIT, algorithm="lloyd").fit_predict(X)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    points = plt.scatter(projected[:, 0], projected[:, 1], c=labels,
                         s=10, alpha=.65, cmap="tab10")
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.title(f"K-Means PCA projection (candidate k={provisional_k})")
    plt.colorbar(points, label="Cluster"); plt.tight_layout()
    plt.savefig(FIGURES / "kmeans_pca_provisional.png", dpi=160); plt.close()

    summary = {
        "customer_count": len(X), "features": FEATURES, "k_values_tested": list(K_VALUES),
        "random_state": RANDOM_STATE, "n_init": N_INIT,
        "silhouette_method": f"Deterministic sample of {sample_size}; DBI and CH use all rows.",
        "provisional_k_by_sampled_silhouette": provisional_k,
        "provisional_k_by_full_silhouette": int(metrics.loc[metrics.silhouette_full.idxmax(), "k"]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "selection_warning": "Do not select final k by silhouette alone; inspect elbow, cluster sizes, profiles and stability."
    }
    (RESULTS / "kmeans_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Customers: {len(X):,}; silhouette sample: {sample_size:,}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Candidate k by sampled silhouette: {provisional_k}")


if __name__ == "__main__":
    main()
