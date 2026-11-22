"""Build customer-level Recency, Frequency and Monetary features.

Run from repository root after Phase 1:
    python src/clean_retail_data.py
    python src/build_rfm.py

The reference date is the day after the latest transaction date in the cleaned
dataset, keeping this fixed-dataset analysis reproducible.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "transactions_clean.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FEATURES = ["Recency", "Frequency", "Monetary"]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run python src/clean_retail_data.py first."
        )
    transactions = pd.read_csv(INPUT, parse_dates=["InvoiceDate"])
    required = {"Invoice", "InvoiceDate", "Quantity", "Price", "Customer ID"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    transactions["Customer ID"] = pd.to_numeric(
        transactions["Customer ID"], errors="coerce"
    )
    transactions["Revenue"] = transactions["Quantity"] * transactions["Price"]
    max_date = transactions["InvoiceDate"].max()
    reference_date = max_date.normalize() + pd.Timedelta(days=1)

    rfm = transactions.groupby("Customer ID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        TransactionLines=("Invoice", "size"),
    ).reset_index()
    rfm["Recency"] = (
        reference_date - rfm["LastPurchaseDate"].dt.normalize()
    ).dt.days
    rfm = rfm[
        ["Customer ID", "Recency", "Frequency", "Monetary",
         "TransactionLines", "LastPurchaseDate"]
    ].sort_values("Customer ID").reset_index(drop=True)

    log_rfm = rfm[FEATURES].copy()
    for feature in FEATURES:
        if (log_rfm[feature] < 0).any():
            raise ValueError(f"Negative values found in {feature}; log1p is invalid.")
        log_rfm[feature] = np.log1p(log_rfm[feature])

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(log_rfm), columns=FEATURES
    )
    scaled.insert(0, "Customer ID", rfm["Customer ID"].values)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(RESULTS / "customer_rfm.csv", index=False)
    scaled.to_csv(RESULTS / "rfm_scaled.csv", index=False)

    raw_stats = rfm[FEATURES].describe(
        percentiles=[.25, .5, .75]
    ).T
    raw_stats["skewness"] = rfm[FEATURES].skew()
    raw_stats["kurtosis"] = rfm[FEATURES].kurtosis()
    raw_stats.to_csv(RESULTS / "rfm_feature_statistics.csv")

    log_stats = log_rfm.describe(
        percentiles=[.25, .5, .75]
    ).T
    log_stats["skewness"] = log_rfm.skew()
    log_stats["kurtosis"] = log_rfm.kurtosis()
    log_stats.to_csv(RESULTS / "rfm_log_feature_statistics.csv")

    summary = {
        "dataset": "UCI Online Retail II",
        "source_transaction_rows": int(len(transactions)),
        "customer_count": int(len(rfm)),
        "reference_date": reference_date.date().isoformat(),
        "latest_transaction_timestamp": max_date.isoformat(),
        "definitions": {
            "Recency": "Days between the fixed reference date and the customer's most recent purchase date; lower is more recent.",
            "Frequency": "Number of distinct invoice IDs per customer, not number of transaction lines.",
            "Monetary": "Sum of Quantity multiplied by unit Price across retained transactions, in GBP.",
        },
        "raw_feature_statistics": {
            feature: {
                "mean": float(rfm[feature].mean()),
                "median": float(rfm[feature].median()),
                "std": float(rfm[feature].std()),
                "min": float(rfm[feature].min()),
                "q1": float(rfm[feature].quantile(.25)),
                "q3": float(rfm[feature].quantile(.75)),
                "max": float(rfm[feature].max()),
                "skewness": float(rfm[feature].skew()),
            } for feature in FEATURES
        },
        "transform": "Apply log1p independently to Recency, Frequency and Monetary, then StandardScaler; Customer ID is excluded from features.",
        "log_feature_skewness": {
            feature: float(log_rfm[feature].skew()) for feature in FEATURES
        },
        "limitations": [
            "Monetary is gross retained sales because cancellation invoices were excluded in Phase 1 rather than netted.",
            "Frequency counts distinct invoice IDs; it does not count product lines.",
            "The reference date is fixed from the dataset maximum date plus one calendar day, not the current date.",
            "RFM values are descriptive customer features, not labels or proof of customer lifetime value.",
        ],
    }
    (RESULTS / "rfm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for feature in FEATURES:
        for values, suffix, label in [
            (rfm[feature], "raw", feature),
            (log_rfm[feature], "log", f"log1p({feature})"),
        ]:
            plt.figure(figsize=(8, 4.5))
            plt.hist(values, bins=50)
            plt.title(f"{label} Distribution")
            plt.xlabel(label)
            plt.ylabel("Customers")
            plt.tight_layout()
            plt.savefig(FIGURES / f"rfm_{feature.lower()}_{suffix}.png", dpi=160)
            plt.close()

    print(f"Customers: {len(rfm):,}")
    print(f"Reference date: {reference_date.date()}")
    print("Raw RFM statistics:")
    print(raw_stats[["mean", "50%", "max", "skewness"]].to_string())
    print("Log1p RFM skewness:")
    print(log_rfm.skew().to_string())
    print("Saved customer RFM, scaled features, statistics, summary and plots.")


if __name__ == "__main__":
    main()
