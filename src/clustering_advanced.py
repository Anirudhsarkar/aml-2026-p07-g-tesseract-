"""Phase 4 (rest): hierarchical clustering, DBSCAN and K-Means stability.

Run from the repository root after Phase 3 and src/clustering.py:
    python src/clustering_advanced.py

Hierarchical: Ward, complete and average linkage cut at k=2..10 (silhouette on
all customers, Davies-Bouldin, Calinski-Harabasz, cophenetic correlation) plus
dendrograms. DBSCAN: k-distance plots, an (eps, min_samples) grid, noise
analysis. Stability: Adjusted Rand Index of K-Means across 80% subsamples.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors

from common import (FEATURES, FIGURES, RANDOM_STATE, RESULTS, fit_kmeans,
                    kmeans_stability, load_rfm)

K_VALUES = list(range(2, 11))
LINKAGES = ["ward", "complete", "average"]
MIN_SAMPLES = [5, 10, 20]
EPS_GRID = [round(float(e), 2) for e in np.arange(0.10, 0.81, 0.05)]
MAX_NOISE_PCT = 15.0
MIN_CLUSTER_SHARE = 0.01
COMPARE_K_RANGE = range(3, 7)


def score_labels(X, labels):
    """Return (silhouette, DB, CH) or NaNs if fewer than two clusters."""
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    return (float(silhouette_score(X, labels)),
            float(davies_bouldin_score(X, labels)),
            float(calinski_harabasz_score(X, labels)))


def run_hierarchical(X, kmeans_labels):
    dists = pdist(X)
    rows, ari_rows, coph = [], [], {}
    linkages = {}
    for method in LINKAGES:
        Z = linkage(X, method=method)
        linkages[method] = Z
        coph[method] = float(cophenet(Z, dists)[0])
        for k in K_VALUES:
            labels = fcluster(Z, k, criterion="maxclust") - 1
            sil, dbi, chi = score_labels(X, labels)
            counts = np.bincount(labels)
            rows.append({
                "linkage": method, "k": k, "clusters_found": int(len(np.unique(labels))),
                "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi,
                "min_cluster_share": float(counts[counts > 0].min() / len(labels)),
                "cophenetic_correlation": coph[method],
            })
            ari_rows.append({
                "linkage": method, "k": k,
                "ari_vs_kmeans": float(adjusted_rand_score(kmeans_labels[k], labels)),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "hierarchical_metrics.csv", index=False)
    pd.DataFrame(ari_rows).to_csv(RESULTS / "hierarchical_vs_kmeans_ari.csv", index=False)

    plt.figure(figsize=(8, 5))
    for method in LINKAGES:
        sub = metrics[metrics.linkage == method]
        plt.plot(sub.k, sub.silhouette, marker="o", label=method)
    plt.xticks(K_VALUES); plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette (all customers)")
    plt.title("Hierarchical Clustering: Silhouette by Linkage")
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / "hierarchical_silhouette.png", dpi=160); plt.close()

    for method in ["ward", "average"]:
        sub = metrics[(metrics.linkage == method) & metrics.k.isin(COMPARE_K_RANGE)]
        cut_k = int(sub.loc[sub.silhouette.idxmax(), "k"])
        Z = linkages[method]
        plt.figure(figsize=(10, 5.5))
        dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True,
                   leaf_rotation=90, leaf_font_size=8, no_labels=False)
        height = (Z[-(cut_k - 1), 2] + Z[-cut_k, 2]) / 2 if cut_k > 1 else Z[-1, 2]
        plt.axhline(height, color="red", linestyle="--", label=f"cut for k={cut_k}")
        plt.title(f"Dendrogram ({method} linkage, last 30 merges)")
        plt.xlabel("Cluster size (or customer index)"); plt.ylabel("Distance")
        plt.legend(); plt.tight_layout()
        plt.savefig(FIGURES / f"dendrogram_{method}.png", dpi=160); plt.close()
    return metrics, coph


def run_dbscan(X, raw):
    nn_tables = {}
    plt.figure(figsize=(8, 5))
    for ms in MIN_SAMPLES:
        distances, _ = NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)
        nn_tables[ms] = np.sort(distances[:, -1])
        plt.plot(nn_tables[ms], label=f"min_samples={ms}")
    plt.xlabel("Customers sorted by distance"); plt.ylabel("Distance to k-th nearest neighbour")
    plt.title("DBSCAN k-distance Plot"); plt.legend(); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIGURES / "dbscan_kdistance.png", dpi=160); plt.close()

    rows = []
    for ms in MIN_SAMPLES:
        for eps in EPS_GRID:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            core = labels >= 0
            n_clusters = int(len(set(labels[core])))
            noise_pct = float((~core).mean() * 100)
            sizes = np.bincount(labels[core]) if core.any() else np.array([0])
            min_share = float(sizes.min() / len(X)) if n_clusters else 0.0
            sil = (float(silhouette_score(X[core], labels[core]))
                   if n_clusters >= 2 else np.nan)
            rows.append({
                "eps": eps, "min_samples": ms, "clusters": n_clusters,
                "noise_percent": noise_pct,
                "largest_cluster_share": float(sizes.max() / len(X)) if n_clusters else 0.0,
                "smallest_cluster_share": min_share, "silhouette_non_noise": sil,
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "dbscan_grid.csv", index=False)

    ok = grid[(grid.clusters >= 2) & (grid.noise_percent <= MAX_NOISE_PCT)
              & (grid.smallest_cluster_share >= MIN_CLUSTER_SHARE)]
    if ok.empty:
        chosen, finding = None, (
            "No DBSCAN setting produced >=2 clusters each holding >=1% of customers "
            f"with <={MAX_NOISE_PCT:.0f}% noise: customer density in log-RFM space is "
            "continuous rather than separated into dense islands.")
        ref_eps = float(np.quantile(nn_tables[10], 0.95)); ref_ms = 10
    else:
        chosen = ok.sort_values("silhouette_non_noise").iloc[-1].to_dict()
        chosen["min_samples"] = int(chosen["min_samples"])
        finding = "Best DBSCAN configuration under the stated constraints."
        ref_eps, ref_ms = float(chosen["eps"]), int(chosen["min_samples"])

    labels = DBSCAN(eps=ref_eps, min_samples=ref_ms).fit_predict(X)
    noise = labels < 0
    profile = raw[FEATURES].assign(group=np.where(noise, "noise", "clustered")).groupby("group").agg(
        customers=("Recency", "size"),
        recency_mean=("Recency", "mean"), recency_median=("Recency", "median"),
        frequency_mean=("Frequency", "mean"), frequency_median=("Frequency", "median"),
        monetary_mean=("Monetary", "mean"), monetary_median=("Monetary", "median"),
    ).reset_index()
    profile.insert(1, "eps", ref_eps); profile.insert(2, "min_samples", ref_ms)
    profile.to_csv(RESULTS / "dbscan_noise_profile.csv", index=False)

    proj = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[~noise, 0], proj[~noise, 1], c=labels[~noise], s=8, cmap="tab10", alpha=.6)
    plt.scatter(proj[noise, 0], proj[noise, 1], c="black", s=10, marker="x", label="noise")
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.title(f"DBSCAN (eps={ref_eps:.2f}, min_samples={ref_ms}) in PCA space")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "dbscan_pca.png", dpi=160); plt.close()
    return grid, chosen, finding, {"eps": ref_eps, "min_samples": ref_ms,
                                    "noise_percent": float(noise.mean() * 100),
                                    "clusters": int(len(set(labels[~noise])))}


def run_stability(X):
    rows = []
    for k in K_VALUES:
        mean_ari, std_ari = kmeans_stability(X, k)
        rows.append({"k": k, "stability_ari_mean": mean_ari, "stability_ari_std": std_ari,
                     "n_bootstrap": 20, "subsample_fraction": 0.8})
    stability = pd.DataFrame(rows)
    stability.to_csv(RESULTS / "kmeans_stability.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.errorbar(stability.k, stability.stability_ari_mean, yerr=stability.stability_ari_std,
                 marker="o", capsize=3)
    plt.xticks(K_VALUES); plt.ylim(0, 1.05); plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean ARI vs full-data clustering")
    plt.title("K-Means Stability (20 subsamples of 80%)")
    plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / "kmeans_stability.png", dpi=160); plt.close()
    return stability


def main():
    raw, _, X = load_rfm()
    FIGURES.mkdir(parents=True, exist_ok=True)

    kmeans_labels = {k: fit_kmeans(X, k).labels_ for k in K_VALUES}
    km = pd.read_csv(RESULTS / "kmeans_metrics.csv") if (RESULTS / "kmeans_metrics.csv").exists() else None

    hier, coph = run_hierarchical(X, kmeans_labels)
    grid, chosen, finding, reference = run_dbscan(X, raw)
    stability = run_stability(X)

    comparison = []
    for k in COMPARE_K_RANGE:
        sil, dbi, chi = score_labels(X, kmeans_labels[k])
        comparison.append({"algorithm": "K-Means", "configuration": f"k={k}",
                           "clusters": k, "silhouette": sil, "davies_bouldin": dbi,
                           "calinski_harabasz": chi, "noise_percent": 0.0})
    for method in LINKAGES:
        sub = hier[(hier.linkage == method) & hier.k.isin(COMPARE_K_RANGE)]
        for _, row in sub.iterrows():
            comparison.append({"algorithm": f"Hierarchical ({method})",
                               "configuration": f"k={int(row.k)}", "clusters": int(row.clusters_found),
                               "silhouette": row.silhouette, "davies_bouldin": row.davies_bouldin,
                               "calinski_harabasz": row.calinski_harabasz, "noise_percent": 0.0})
    if chosen is not None:
        core_note = f"eps={chosen['eps']}, min_samples={chosen['min_samples']}"
        comparison.append({"algorithm": "DBSCAN", "configuration": core_note,
                           "clusters": int(chosen["clusters"]),
                           "silhouette": chosen["silhouette_non_noise"], "davies_bouldin": np.nan,
                           "calinski_harabasz": np.nan, "noise_percent": chosen["noise_percent"]})
    pd.DataFrame(comparison).to_csv(RESULTS / "algorithm_comparison.csv", index=False)

    summary = {
        "customer_count": int(len(X)), "random_state": RANDOM_STATE,
        "hierarchical": {
            "linkages": LINKAGES, "cophenetic_correlation": coph,
            "best_k_by_silhouette_in_3_to_6": {
                m: int(hier[(hier.linkage == m) & hier.k.isin(COMPARE_K_RANGE)]
                       .sort_values("silhouette").iloc[-1].k) for m in LINKAGES},
        },
        "dbscan": {
            "grid_eps": EPS_GRID, "grid_min_samples": MIN_SAMPLES,
            "constraints": {"max_noise_percent": MAX_NOISE_PCT,
                            "min_cluster_share": MIN_CLUSTER_SHARE, "min_clusters": 2},
            "chosen": chosen, "finding": finding, "reference_configuration_for_noise_analysis": reference,
        },
        "kmeans_stability_ari_mean": {int(r.k): r.stability_ari_mean for r in stability.itertuples()},
        "notes": [
            "Silhouette is computed on all customers for K-Means and hierarchical clusterings.",
            "DBSCAN silhouette excludes noise points and is therefore not directly comparable.",
            "Do not select the final k by silhouette alone; see src/segments.py for the documented selection rule.",
        ],
    }
    if km is not None and "silhouette_full" in km:
        summary["kmeans_k_by_full_silhouette"] = int(km.loc[km.silhouette_full.idxmax(), "k"])
    (RESULTS / "clustering_comparison.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Cophenetic correlation:", {m: round(v, 3) for m, v in coph.items()})
    print("Hierarchical best k (3-6):", summary["hierarchical"]["best_k_by_silhouette_in_3_to_6"])
    print("DBSCAN:", finding)
    print("K-Means stability (ARI):", {k: round(v, 3) for k, v in summary["kmeans_stability_ari_mean"].items()})


if __name__ == "__main__":
    main()
