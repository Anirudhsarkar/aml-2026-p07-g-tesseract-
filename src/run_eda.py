"""Phase 2 EDA for cleaned UCI Online Retail II transactions.

Outputs (results/): monthly revenue, top products, country analysis, missing
values, outlier diagnostics, customer purchasing patterns, and figures.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "transactions_clean.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
INVOICE_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10**9, "11+")]


def most_common(series: pd.Series):
    modes = series.dropna().value_counts()
    return modes.index[0] if len(modes) else None


def purchasing_patterns(data: pd.DataFrame) -> dict:
    """Customer-level purchasing behaviour; writes tables and figures."""
    invoices = data.groupby(["Customer ID", "Invoice"]).agg(
        invoice_value=("Revenue", "sum"), invoice_lines=("Revenue", "size"),
        invoice_units=("Quantity", "sum"), invoice_date=("InvoiceDate", "min"),
    ).reset_index()
    per_customer = invoices.groupby("Customer ID").agg(
        invoices=("Invoice", "nunique"),
        avg_invoice_value=("invoice_value", "mean"),
        avg_lines_per_invoice=("invoice_lines", "mean"),
        avg_units_per_invoice=("invoice_units", "mean"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )
    per_customer["distinct_products"] = data.groupby("Customer ID")["StockCode"].nunique()
    per_customer["active_days"] = (
        data.assign(day=data["InvoiceDate"].dt.normalize())
        .groupby("Customer ID")["day"].nunique()
    )
    span_days = (per_customer["last_purchase"] - per_customer["first_purchase"]).dt.days
    per_customer["avg_days_between_invoices"] = np.where(
        per_customer["invoices"] > 1, span_days / (per_customer["invoices"] - 1).clip(lower=1), np.nan
    )
    per_customer.reset_index().to_csv(RESULTS_DIR / "customer_purchasing_patterns.csv", index=False)

    buckets = []
    for low, high, label in INVOICE_BUCKETS:
        mask = per_customer["invoices"].between(low, high)
        buckets.append({
            "invoice_count_bucket": label,
            "customers": int(mask.sum()),
            "customer_share": float(mask.mean()),
            "revenue_gbp": float(data[data["Customer ID"].isin(per_customer.index[mask])]["Revenue"].sum()),
        })
    buckets = pd.DataFrame(buckets)
    buckets["revenue_share"] = buckets["revenue_gbp"] / buckets["revenue_gbp"].sum()
    buckets.to_csv(RESULTS_DIR / "purchase_frequency_buckets.csv", index=False)

    activity = data.assign(
        weekday=data["InvoiceDate"].dt.day_name(), hour=data["InvoiceDate"].dt.hour
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_weekday = activity.groupby("weekday").agg(
        revenue_gbp=("Revenue", "sum"), invoices=("Invoice", "nunique")
    ).reindex(weekday_order).fillna(0).reset_index()
    by_hour = activity.groupby("hour").agg(
        revenue_gbp=("Revenue", "sum"), invoices=("Invoice", "nunique")
    ).reset_index()
    by_weekday.to_csv(RESULTS_DIR / "weekday_activity.csv", index=False)
    by_hour.to_csv(RESULTS_DIR / "hourly_activity.csv", index=False)

    first_month = per_customer["first_purchase"].dt.to_period("M")
    invoices["month"] = invoices["invoice_date"].dt.to_period("M")
    invoices["first_month"] = invoices["Customer ID"].map(first_month)
    monthly_customers = invoices.groupby("month").agg(
        active_customers=("Customer ID", "nunique")).reset_index()
    new_per_month = (invoices[invoices["month"] == invoices["first_month"]]
                     .groupby("month")["Customer ID"].nunique())
    monthly_customers["new_customers"] = (
        monthly_customers["month"].map(new_per_month).fillna(0).astype(int))
    monthly_customers["returning_customers"] = (
        monthly_customers["active_customers"] - monthly_customers["new_customers"]
    )
    monthly_customers["month"] = monthly_customers["month"].astype(str)
    monthly_customers.to_csv(RESULTS_DIR / "monthly_new_vs_returning.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.hist(per_customer["invoices"].clip(upper=40), bins=40)
    plt.xlabel("Invoices per customer (clipped at 40)")
    plt.ylabel("Customers")
    plt.title("Purchase Frequency Distribution")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "invoices_per_customer.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(by_weekday["weekday"], by_weekday["revenue_gbp"])
    plt.xticks(rotation=30, ha="right"); plt.ylabel("Revenue (£)")
    plt.title("Revenue by Weekday")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "weekday_revenue.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(monthly_customers["month"], monthly_customers["returning_customers"], label="Returning")
    plt.bar(monthly_customers["month"], monthly_customers["new_customers"],
            bottom=monthly_customers["returning_customers"], label="New")
    plt.xticks(rotation=60, ha="right"); plt.ylabel("Active customers")
    plt.title("Active Customers per Month: New vs Returning"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "new_vs_returning_customers.png", dpi=160); plt.close()

    one_time = per_customer["invoices"] == 1
    return {
        "customers": int(len(per_customer)),
        "one_time_customers": int(one_time.sum()),
        "one_time_share": float(one_time.mean()),
        "repeat_customers": int((~one_time).sum()),
        "median_invoices_per_customer": float(per_customer["invoices"].median()),
        "median_avg_invoice_value_gbp": float(per_customer["avg_invoice_value"].median()),
        "median_lines_per_invoice": float(per_customer["avg_lines_per_invoice"].median()),
        "median_days_between_invoices_repeat_customers": float(
            per_customer["avg_days_between_invoices"].median()),
        "top_10_percent_customers_revenue_share": float(
            data.groupby("Customer ID")["Revenue"].sum().sort_values(ascending=False)
            .head(max(1, int(len(per_customer) * 0.10))).sum() / data["Revenue"].sum()),
        "busiest_weekday_by_revenue": str(by_weekday.loc[by_weekday["revenue_gbp"].idxmax(), "weekday"]),
        "busiest_hour_by_revenue": int(by_hour.loc[by_hour["revenue_gbp"].idxmax(), "hour"]),
    }


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

    date_min, date_max = data["InvoiceDate"].min(), data["InvoiceDate"].max()
    monthly = data.groupby("Month", as_index=False).agg(
        revenue_gbp=("Revenue", "sum"),
        transactions=("Invoice", "nunique"),
        units_sold=("Quantity", "sum"),
    )
    # A month is partial when the data starts after day 1 or ends before the last day.
    month_start = pd.PeriodIndex(monthly["Month"], freq="M")
    monthly["is_partial_month"] = [
        bool(
            (p == date_min.to_period("M") and date_min.day > 1)
            or (p == date_max.to_period("M") and date_max.day < p.days_in_month)
        )
        for p in month_start
    ]
    # Group by StockCode so that products with several description spellings are not split.
    descriptions = data.groupby("StockCode")["Description"].agg(most_common)
    products = data.groupby("StockCode").agg(
        units_sold=("Quantity", "sum"),
        revenue_gbp=("Revenue", "sum"),
        transactions=("Invoice", "nunique"),
    ).reset_index()
    products.insert(1, "Description", products["StockCode"].map(descriptions))
    products = products.sort_values("revenue_gbp", ascending=False)
    countries = data.groupby("Country").agg(
        transactions=("Invoice", "nunique"),
        customers=("Customer ID", "nunique"),
        revenue_gbp=("Revenue", "sum"),
        units_sold=("Quantity", "sum"),
    ).reset_index().sort_values("revenue_gbp", ascending=False)
    missing = data.drop(columns=["Month"]).isna().sum().rename_axis("column").reset_index(name="missing_values")

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
    full = monthly[~monthly["is_partial_month"]]
    plt.plot(full["Month"], full["revenue_gbp"], marker="o", label="Full months")
    partial = monthly[monthly["is_partial_month"]]
    if len(partial):
        plt.scatter(partial["Month"], partial["revenue_gbp"], marker="x", color="red",
                    s=70, zorder=3, label="Partial month (incomplete data)")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Revenue (£)")
    plt.xlabel("Month")
    plt.title("Monthly Revenue")
    plt.legend()
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

    patterns = purchasing_patterns(data)
    full_months = monthly[~monthly["is_partial_month"]]
    peak_pool = full_months if len(full_months) else monthly

    summary = {
        "dataset": "UCI Online Retail II",
        "cleaned_rows": int(len(data)),
        "unique_customers": int(data["Customer ID"].nunique()),
        "unique_invoices": int(data["Invoice"].nunique()),
        "unique_products_stock_codes": int(data["StockCode"].nunique()),
        "countries": int(data["Country"].nunique()),
        "date_min": date_min.isoformat(),
        "date_max": date_max.isoformat(),
        "revenue_currency": "GBP",
        "total_gross_revenue_gbp": float(data["Revenue"].sum()),
        "monthly_revenue_peak": peak_pool.loc[peak_pool["revenue_gbp"].idxmax()].to_dict(),
        "monthly_revenue_low": peak_pool.loc[peak_pool["revenue_gbp"].idxmin()].to_dict(),
        "partial_months": monthly.loc[monthly["is_partial_month"], "Month"].tolist(),
        "top_product_by_revenue": products.iloc[0].to_dict(),
        "top_country_by_revenue": countries.iloc[0].to_dict(),
        "cleaned_data_missing_values": {
            str(k): int(v) for k, v in data.drop(columns=["Month"]).isna().sum().items()},
        "outlier_method": "Tukey upper fence: Q3 + 1.5 * IQR; flags are diagnostic only, not deleted",
        "outlier_summary": outlier_rows,
        "purchasing_patterns": patterns,
        "interpretation_notes": [
            "Revenue is Quantity multiplied by unit Price and is reported in GBP.",
            "The analysis uses Phase 1 cleaned data; excluded cancellations are not netted against original purchases.",
            "Products are grouped by StockCode; the most frequent description is shown.",
            "Peak/low month statistics exclude partial months (first/last month of the data if incomplete).",
            "Inspect single-invoice extreme orders (very large quantity) before treating them as representative demand; they may have been reversed by a cancellation.",
            "The United Kingdom dominates recorded revenue; interpret country totals in light of dataset transaction coverage.",
            "IQR fences flag potential outliers but do not prove data errors; observations are retained pending domain review.",
        ],
    }
    with (RESULTS_DIR / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    print(f"Rows: {len(data):,}")
    print(f"Customers: {data['Customer ID'].nunique():,}")
    print(f"Revenue: £{data['Revenue'].sum():,.2f}")
    print(f"One-time customers: {patterns['one_time_share']:.1%}")
    print("EDA tables, summary and figures saved under results/.")


if __name__ == "__main__":
    main()
