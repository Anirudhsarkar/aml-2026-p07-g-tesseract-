"""
Phase 1: Load and clean the UCI Online Retail II workbook.

Expected input:
    data/raw/online_retail_II.xlsx

Run from the repository root:
    python src/clean_retail_data.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Explicit non-merchandise / service / adjustment codes used in this Phase 1 run.
NON_PRODUCT_CODES = {
    "B", "C2", "C3", "CRUK", "D", "DOT", "GIFT", "M", "m", "POST", "S"
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Download the UCI Online Retail II workbook and place it at "
            "data/raw/online_retail_II.xlsx"
        )

    sheets = pd.read_excel(INPUT_FILE, sheet_name=None, engine="openpyxl")
    frames = list(sheets.values())
    raw = pd.concat(frames, ignore_index=True)
    funnel = [{"stage": "Loaded from both sheets", "rows_remaining": int(len(raw))}]

    data = raw.drop_duplicates().copy()
    funnel.append({"stage": "After exact duplicate removal", "rows_remaining": int(len(data))})

    data = data[data["Customer ID"].notna()].copy()
    funnel.append({"stage": "After removing missing Customer ID", "rows_remaining": int(len(data))})

    data = data[~data["Invoice"].astype(str).str.startswith("C")].copy()
    funnel.append({"stage": "After removing cancellation invoices", "rows_remaining": int(len(data))})

    data = data[~data["StockCode"].astype(str).isin(NON_PRODUCT_CODES)].copy()
    funnel.append({"stage": "After removing non-product/service stock codes", "rows_remaining": int(len(data))})

    data = data[data["Description"].notna()].copy()
    funnel.append({"stage": "After removing missing descriptions", "rows_remaining": int(len(data))})

    data = data[data["Quantity"] > 0].copy()
    funnel.append({"stage": "After removing non-positive quantities", "rows_remaining": int(len(data))})

    data = data[data["Price"] > 0].copy()
    funnel.append({"stage": "After removing non-positive prices", "rows_remaining": int(len(data))})

    # Customer IDs are identifiers, not continuous measurements.
    data["Customer ID"] = data["Customer ID"].astype("Int64")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)

    funnel_df = pd.DataFrame(funnel)
    funnel_df["rows_removed"] = funnel_df["rows_remaining"].shift(1) - funnel_df["rows_remaining"]
    funnel_df.loc[0, "rows_removed"] = 0
    funnel_df.to_csv(RESULTS_DIR / "cleaning_funnel.csv", index=False)

    summary = {
        "dataset": "UCI Online Retail II",
        "source_file": INPUT_FILE.name,
        "sheets": {
            name: {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
            for name, frame in sheets.items()
        },
        "combined_rows": int(len(raw)),
        "combined_columns": [str(column) for column in raw.columns],
        "missing_values_before_cleaning": {
            str(k): int(v) for k, v in raw.isna().sum().items()
        },
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "cancellation_invoice_rows_before_cleaning": int(
            raw["Invoice"].astype(str).str.startswith("C").sum()
        ),
        "cleaning_rules": [
            "Remove exact duplicate transaction rows.",
            "Remove rows without Customer ID because customer-level RFM requires customer identity.",
            "Remove cancellation invoices whose Invoice value starts with C.",
            "Remove non-product/service stock codes: " + ", ".join(sorted(NON_PRODUCT_CODES)) + ".",
            "Remove rows with missing Description.",
            "Remove rows with Quantity <= 0.",
            "Remove rows with Price <= 0."
        ],
        "cleaned_rows": int(len(data)),
        "cleaned_columns": [str(column) for column in data.columns],
        "cleaning_retention_percent": round(len(data) / len(raw) * 100, 3),
        "notes": [
            "Cleaning funnel stages are sequential; rows_removed at each stage are incremental.",
            "No upper-tail quantity or price outlier capping/removal was applied. Extreme positive values should be inspected during EDA rather than silently discarded.",
            "Cancellation invoices are excluded, not netted against original purchases. The team should state this choice in the report and revisit it if a return-adjusted RFM definition is required.",
            "Non-product/service codes were excluded by an explicit list; alphabetic suffixes on ordinary stock codes were retained because many are valid product identifiers."
        ]
    }
    with open(RESULTS_DIR / "cleaning_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Combined rows: {len(raw):,}")
    print(f"Cleaned rows:  {len(data):,}")
    print(f"Retention:     {len(data) / len(raw) * 100:.2f}%")
    print("Saved cleaned data, cleaning funnel, and summary.")


if __name__ == "__main__":
    main()
