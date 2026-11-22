"""Phase 2 EDA for cleaned UCI Online Retail II transactions."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "transactions_clean.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {INPUT_FILE}. "
            "Run src/clean_retail_data.py first."
        )

    data = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])
    data["Revenue"] = data["Quantity"] * data["Price"]
    data["Month"] = data["InvoiceDate"].dt.to_period("M").astype(str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    monthly = data.groupby("Month", as_index=False).agg(
        revenue_gbp=("Revenue", "sum"),
        transactions=("Invoice", "nunique"),
        units_sold=("Quantity", "sum"),
    )
    products = data.groupby(["StockCode", "Description"], dropna=False).agg(
        units_sold=("Quantity", "sum"),
        revenue_gbp=("Revenue", "sum"),
        transactions=("Invoice", "nunique"),
    ).reset_index().sort_values("revenue_gbp", ascending=False)
    countries = data.groupby("Country").agg(
        transactions=("Invoice", "nunique"),
        customers=("Customer ID", "nunique"),
        revenue_gbp=("Revenue", "sum"),
        units_sold=("Quantity", "sum"),
    ).reset_index().sort_values("revenue_gbp", ascending=False)
    missing = data.isna().sum().rename_axis("column").reset_index(name="missing_values")

    outlier_rows = []
    for feature in ["Quantity", "Price", "Revenue"]:
        q1 = float(data[feature].quantile(0.25))
        q3 = float(data[feature].quantile(0.75))
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        count = int((data[feature] > upper).sum())
        outlier_rows.append({
            "feature": feature, "q1": q1, "q3": q3, "iqr": iqr,
            "upper_iqr_fence": upper, "values_above_upper_fence": count,
            "percent_above_upper_fence": float(count / len(data) * 100),
            "maximum": float(data[feature].max()),
        })

    monthly.to_csv(RESULTS_DIR / "monthly_revenue.csv", index=False)
    products.head(10).to_csv(RESULTS_DIR / "top_10_products.csv", index=False)
    countries.to_csv(RESULTS_DIR / "country_analysis.csv", index=False)
    missing.to_csv(RESULTS_DIR / "missing_values_cleaned.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(RESULTS_DIR / "outlier_summary.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["Month"], monthly["revenue_gbp"], marker="o")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Revenue (£)")
    plt.xlabel("Month")
    plt.title("Monthly Revenue")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_revenue.png", dpi=160)
    plt.close()

    top10 = products.head(10).sort_values("revenue_gbp")
    plt.figure(figsize=(10, 6))
    plt.barh(top10["Description"].astype(str), top10["revenue_gbp"])
    plt.xlabel("Revenue (£)")
    plt.title("Top 10 Products by Revenue")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_10_products.png", dpi=160)
    plt.close()

    top15 = countries.head(15).sort_values("revenue_gbp")
    plt.figure(figsize=(10, 6))
    plt.barh(top15["Country"], top15["revenue_gbp"])
    plt.xlabel("Revenue (£)")
    plt.title("Top 15 Countries by Revenue")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "country_revenue_top15.png", dpi=160)
    plt.close()

    summary = {
        "dataset": "UCI Online Retail II",
        "cleaned_rows": int(len(data)),
        "unique_customers": int(data["Customer ID"].nunique()),
        "unique_invoices": int(data["Invoice"].nunique()),
        "unique_products_stock_codes": int(data["StockCode"].nunique()),
        "countries": int(data["Country"].nunique()),
        "date_min": data["InvoiceDate"].min().isoformat(),
        "date_max": data["InvoiceDate"].max().isoformat(),
        "revenue_currency": "GBP",
        "total_gross_revenue_gbp": float(data["Revenue"].sum()),
        "monthly_revenue_peak": monthly.loc[monthly["revenue_gbp"].idxmax()].to_dict(),
        "monthly_revenue_low": monthly.loc[monthly["revenue_gbp"].idxmin()].to_dict(),
        "top_product_by_revenue": products.iloc[0].to_dict(),
        "top_country_by_revenue": countries.iloc[0].to_dict(),
        "cleaned_data_missing_values": {str(k): int(v) for k, v in data.isna().sum().items()},
        "outlier_method": "Tukey upper fence: Q3 + 1.5 * IQR; flags are diagnostic only, not deleted",
        "outlier_summary": outlier_rows,
        "interpretation_notes": [
            "Revenue is Quantity multiplied by unit Price and is reported in GBP.",
            "The analysis uses Phase 1 cleaned data; excluded cancellations are not netted against original purchases.",
            "The PAPER CRAFT, LITTLE BIRDIE product has an unusually large one-invoice quantity/revenue; investigate before treating it as representative demand.",
            "The United Kingdom dominates recorded revenue; interpret country totals in light of dataset transaction coverage.",
            "IQR fences flag potential outliers but do not prove data errors; observations are retained pending domain review.",
        ],
    }
    with (RESULTS_DIR / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Rows: {len(data):,}")
    print(f"Customers: {data['Customer ID'].nunique():,}")
    print(f"Revenue: £{data['Revenue'].sum():,.2f}")
    print("EDA tables, summary and figures saved under results/.")


if __name__ == "__main__":
    main()
