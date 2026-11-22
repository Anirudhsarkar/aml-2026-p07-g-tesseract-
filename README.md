# P07 — Customer Segmentation for an Online Retailer

**Course:** Applied Machine Learning (CSAI2017P), Autumn 2026  
**Theme:** P07 — Retail Segmentation  
**Group:** Team Tesseract

## Project question
Which genuinely distinct customer groups exist in online retail transactions, and can a model assign a new customer to a segment using only their early purchase behaviour?

## Dataset
Use [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Download the dataset ZIP in a browser, extract the workbook, and place it at `data/raw/online_retail_II.xlsx`. See [dataset/README.md](dataset/README.md) for the download and placement instructions.

**Do not commit the raw ZIP or workbook.** The workbook is approximately 43.5 MB and remains local; the repository tracks the source, instructions, scripts and analysis outputs.

## Setup
Recommended Python 3.11 or 3.12. Use a consistent Python version across the team.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Phase 1 — Dataset loading and cleaning

The workbook was inspected and both sheets were combined.

| Source sheet | Rows | Columns |
|---|---:|---:|
| Year 2009–2010 | 525,461 | 8 |
| Year 2010–2011 | 541,910 | 8 |
| **Combined** | **1,067,371** | **8** |

The current baseline retains **776,672 transactions (72.765%)** after sequential cleaning. See [cleaning funnel](results/cleaning_funnel.csv) and [cleaning summary](results/cleaning_summary.json).

Cleaning decisions:
- Remove exact duplicate rows.
- Remove transactions without Customer ID because customer-level RFM requires customer identity.
- Exclude cancellation invoices whose Invoice value begins with `C`; these are not netted against original purchases.
- Exclude the explicit non-product/service StockCode list in the cleaning script.
- Remove rows with invalid core fields and non-positive quantities or prices.
- Do not automatically cap or remove extreme positive quantity/price values; inspect them during EDA.

Run from the repository root:

```bash
python src/clean_retail_data.py
```

This creates the local file `data/processed/transactions_clean.csv` and the Phase 1 diagnostics. The raw and processed transaction data are excluded from Git.

## Phase 2 — Exploratory Data Analysis

The EDA script calculates monthly revenue, product performance, country-level activity, missing-value checks and IQR-based outlier diagnostics.

```bash
python src/run_eda.py
```

The script reads `data/processed/transactions_clean.csv` and writes:
- [Monthly revenue](results/monthly_revenue.csv)
- [Top 10 products by revenue](results/top_10_products.csv)
- [Country analysis](results/country_analysis.csv)
- [Missing values after cleaning](results/missing_values_cleaned.csv)
- [Outlier summary](results/outlier_summary.csv)
- [EDA summary](results/eda_summary.json)
- Three charts under `results/figures/` (monthly revenue, top products and top countries).

### Initial EDA findings

- Cleaned transactions: **776,672**
- Unique customers: **5,861**
- Unique invoices: **36,669**
- Distinct stock codes: **4,626**
- Countries: **41**
- Period covered: **1 December 2009 – 9 December 2011**
- Gross transaction revenue: **£17,073,528.31**
- Highest-revenue month: **November 2010 (£1,156,008.39)**
- Highest-revenue country: **United Kingdom (£14,291,644.57)**

The highest-revenue product is **REGENCY CAKESTAND 3 TIER (£277,656.25)**. A separate product, **PAPER CRAFT, LITTLE BIRDIE**, has 80,995 units recorded in one invoice and revenue of £168,469.60. This unusual observation is flagged for investigation, not automatically removed.

Outliers are flagged using the Tukey upper fence (Q3 + 1.5 × IQR). These flags are diagnostic only; no upper-tail observations were removed in Phase 2.

## Planned analysis
1. **RFM:** customer-level Recency, Frequency and Monetary values with a fixed reference date; inspect skew and scale features.
2. **Clustering:** K-Means for k=2–10 (elbow/inertia and silhouette), hierarchical agglomerative clustering (dendrogram and at least two linkages), and DBSCAN (k-distance plot/noise analysis).
3. **Interpretation:** cluster sizes and mean RFM profiles, business-facing segment labels/recommendations, and PCA 2D projection.
4. **Classification:** Logistic Regression plus at least one tree-based classifier predicting segment from early-purchase behaviour. Avoid leakage from lifetime RFM or target-defining information.
5. **Evaluation:** accuracy, macro-F1, per-class recall, confusion matrix, and `results/metrics.csv`.
6. **Report:** 4–6 page report, reproducible commands and honest limitations.

## Team
| Member | Enrolment number | GitHub username |
|---|---:|---|
| Aryan Kumar Maurya | 590022160 | TODO |
| Anirudh Sarkar | 590025468 | TODO |
| Devam Yadav | 590022492 | TODO |
| Aniket Malik | 590024567 | TODO |
| Mansha | 590021940 | TODO |
| Shatakshi | 590021766 | TODO |
| Saisradha Das | 590029397 | TODO |
| Pragati | 590028725 | TODO |

## Contributions
Agree and record actual responsibilities here. Every member should understand the code and decisions they contribute.

## Headline results
**Clustering and classification metrics pending.** Cleaning retention and EDA statistics are data-preparation/descriptive results, not model-performance metrics.

## AI and external resources
Acknowledge AI tools and external resources used. Verify and understand all retained code; the individual assessment tests understanding.

## Course milestones
- `checkpoint-1`: actual data loaded, shape/head/chart, RFM table, and one baseline K-Means run; create the annotated tag when genuinely complete.
- `checkpoint-2`: full clustering pipeline and initial results; tag only when genuinely complete.
- Final: analysis, evaluation, report, README and `REPRODUCIBILITY.md`.

## Current status
**Phase 1 completed:** dataset loaded and cleaned, cleaning funnel and summary recorded.  
**Phase 2 completed:** monthly revenue, product, country, missing-value and outlier analyses generated from the Phase 1 cleaned transactions. RFM construction, clustering, classification and final report remain pending.
