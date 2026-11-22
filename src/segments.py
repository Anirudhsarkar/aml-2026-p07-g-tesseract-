"""Phase 5: choose the final k, name the segments, PCA plot, recommendations.

Run from the repository root after src/clustering_advanced.py:
    python src/segments.py            # k chosen by the documented rule
    python src/segments.py --k 4      # override the rule

Selection rule (documented, deliberately simple and overridable): among
k = K_MIN..K_MAX, keep k whose smallest segment holds >= MIN_SHARE of customers
and whose bootstrap stability (ARI) is >= MIN_ARI; choose the highest silhouette
among them. K-Means with k=2 may have the best silhouette but only separates
active from lapsed customers, which is too coarse to act on.

Segment names come from heuristic thresholds on the standardized log-RFM
centroids. They are a starting point and must be reviewed by the team.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from common import (FEATURES, FIGURES, RANDOM_STATE, RESULTS, fit_kmeans,
                    kmeans_stability, load_rfm)

K_MIN, K_MAX = 3, 6
MIN_SHARE = 0.05
MIN_ARI = 0.75

RECOMMENDATIONS = {
    "Champions": (
        "Most recent, most frequent, highest spend.",
        "VIP/early-access programme; loyalty rewards; referral incentives; avoid blanket discounts."),
    "Loyal High-Value": (
        "Frequent, high-spending and still reasonably active.",
        "Upsell and bundle/wholesale offers; replenishment reminders; seasonal previews."),
    "At-Risk High-Value": (
        "Previously frequent, high spend, but have gone quiet.",
        "Priority win-back: personal outreach, offers on previously bought categories, feedback survey."),
    "Recent Customers": (
        "Bought recently but with few orders so far.",
        "Onboarding emails, second-purchase incentive, product recommendations to build the habit."),
    "Lost / Dormant": (
        "Long inactive with low order count.",
        "Low-cost reactivation email series; suppress from paid campaigns if no response."),
    "Occasional / Low-Value": (
        "Infrequent, low-spend, middling recency.",
        "Seasonal campaigns, free-shipping thresholds and bundles to raise basket size; cross-sell."),
}


DOWNGRADE = {"Champions": "Loyal High-Value"}


def name_segment(recency_z: float, frequency_z: float, monetary_z: float) -> str:
    """Heuristic naming from centroid z-scores (recency flipped: higher = more recent)."""
    r = -recency_z
    if r > 0.3 and frequency_z > 0.5 and monetary_z > 0.5:
        return "Champions"
    if frequency_z > 0.2 and monetary_z > 0.2:
        return "Loyal High-Value" if r > -0.3 else "At-Risk High-Value"
    if r > 0.3 and frequency_z <= 0.2:
        return "Recent Customers"
    if r < -0.3 and frequency_z <= 0.2:
        return "Lost / Dormant"
    return "Occasional / Low-Value"


def select_k(X):
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        labels = fit_kmeans(X, k).labels_
        ari_mean, ari_std = kmeans_stability(X, k)
        rows.append({
            "k": k, "silhouette_full": float(silhouette_score(X, labels)),
            "min_cluster_share": float(np.bincount(labels).min() / len(labels)),
            "stability_ari_mean": ari_mean, "stability_ari_std": ari_std,
        })
    table = pd.DataFrame(rows)
    ok = table[(table.min_cluster_share >= MIN_SHARE) & (table.stability_ari_mean >= MIN_ARI)]
    if ok.empty:
        chosen = int(table.sort_values("stability_ari_mean").iloc[-1].k)
        rule = "fallback: no k met the constraints; most stable k chosen"
    else:
        chosen = int(ok.sort_values("silhouette_full").iloc[-1].k)
        rule = (f"highest silhouette among k in {K_MIN}..{K_MAX} with smallest segment >= "
                f"{MIN_SHARE:.0%} of customers and stability ARI >= {MIN_ARI}")
    table["meets_constraints"] = (table.min_cluster_share >= MIN_SHARE) & (table.stability_ari_mean >= MIN_ARI)
    table["chosen"] = table.k == chosen
    table.to_csv(RESULTS / "k_selection.csv", index=False)
    return chosen, rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None, help="override the selection rule")
    args = parser.parse_args()

    raw, scaled, X = load_rfm()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.k is None:
        k, rule = select_k(X)
    else:
        k, rule = args.k, "manual override via --k"
        select_k(X)
    model = fit_kmeans(X, k)
    labels = model.labels_

    frame = raw[["Customer ID", *FEATURES]].copy()
    frame["Cluster"] = labels
    centroids = pd.DataFrame(X).groupby(labels).mean()
    centroids.columns = FEATURES
    names = {c: name_segment(*centroids.loc[c, FEATURES]) for c in centroids.index}

    # Rank clusters by median monetary value (segment 0 = highest value), make names unique.
    order = frame.groupby("Cluster")["Monetary"].median().sort_values(ascending=False).index.tolist()
    seen, final_names, segment_id, base_names = {}, {}, {}, {}
    for new_id, cluster in enumerate(order):
        base = names[cluster]
        if base in seen and base in DOWNGRADE and DOWNGRADE[base] not in seen:
            base = DOWNGRADE[base]  # only the highest-value cluster keeps the top label
        base_names[cluster] = base
        seen[base] = seen.get(base, 0) + 1
        final_names[cluster] = base if seen[base] == 1 else f"{base} ({seen[base]})"
        segment_id[cluster] = new_id
    frame["Segment"] = frame["Cluster"].map(segment_id)
    frame["SegmentName"] = frame["Cluster"].map(final_names)

    total_revenue = frame["Monetary"].sum()
    profile_rows = []
    for cluster in order:
        sub = frame[frame.Cluster == cluster]
        description, actions = RECOMMENDATIONS[base_names[cluster]]
        profile_rows.append({
            "Segment": segment_id[cluster], "SegmentName": final_names[cluster],
            "customers": int(len(sub)), "customer_share": len(sub) / len(frame),
            "revenue_gbp": float(sub.Monetary.sum()),
            "revenue_share": float(sub.Monetary.sum() / total_revenue),
            "recency_mean": sub.Recency.mean(), "recency_median": sub.Recency.median(),
            "frequency_mean": sub.Frequency.mean(), "frequency_median": sub.Frequency.median(),
            "monetary_mean": sub.Monetary.mean(), "monetary_median": sub.Monetary.median(),
            "centroid_z_recency": centroids.loc[cluster, "Recency"],
            "centroid_z_frequency": centroids.loc[cluster, "Frequency"],
            "centroid_z_monetary": centroids.loc[cluster, "Monetary"],
            "description": description, "recommended_actions": actions,
        })
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(RESULTS / "segment_profiles.csv", index=False)
    profiles[["Segment", "SegmentName", "description", "recommended_actions"]].to_csv(
        RESULTS / "segment_recommendations.csv", index=False)
    frame[["Customer ID", "Segment", "SegmentName", *FEATURES]].to_csv(
        RESULTS / "customer_segments.csv", index=False)

    # Figures
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(X)
    plt.figure(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10")
    for _, row in profiles.iterrows():
        mask = (frame.Segment == row.Segment).to_numpy()
        plt.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=.65, color=cmap(int(row.Segment)),
                    label=f"{row.SegmentName} (n={row.customers:,})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"Customer Segments in PCA Space (K-Means, k={k})")
    plt.legend(markerscale=2, fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / "segments_pca.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    width = 0.25
    xs = np.arange(len(profiles))
    for i, (col, lab) in enumerate([("centroid_z_recency", "Recency (z, high = older)"),
                                    ("centroid_z_frequency", "Frequency (z)"),
                                    ("centroid_z_monetary", "Monetary (z)")]):
        plt.bar(xs + (i - 1) * width, profiles[col], width, label=lab)
    plt.xticks(xs, profiles.SegmentName, rotation=20, ha="right"); plt.axhline(0, color="k", lw=.8)
    plt.ylabel("Standardized log-RFM centroid"); plt.title("Segment RFM Profile"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "segment_rfm_profile.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(xs - 0.2, profiles.customer_share * 100, 0.4, label="% of customers")
    plt.bar(xs + 0.2, profiles.revenue_share * 100, 0.4, label="% of revenue")
    plt.xticks(xs, profiles.SegmentName, rotation=20, ha="right")
    plt.ylabel("Share (%)"); plt.title("Customer Share vs Revenue Share by Segment"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "segment_share_vs_revenue.png", dpi=160); plt.close()

    summary = {
        "final_k": k, "selection_rule": rule, "k_range_considered": [K_MIN, K_MAX],
        "constraints": {"min_cluster_share": MIN_SHARE, "min_stability_ari": MIN_ARI},
        "random_state": RANDOM_STATE,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "segments": profiles[["Segment", "SegmentName", "customers", "customer_share",
                              "revenue_share"]].to_dict("records"),
        "caveats": [
            "Segment names are heuristic labels from centroid thresholds; review them against the profiles.",
            "Monetary is gross sales (cancellations excluded, not netted), so very high values can include reversed orders.",
            "PCA is a 2D projection of a 3-feature space; overlap in the plot does not imply poor clustering in 3D.",
        ],
    }
    (RESULTS / "segment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Final k = {k} ({rule})")
    print(profiles[["Segment", "SegmentName", "customers", "customer_share", "revenue_share"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
