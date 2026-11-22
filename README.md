# P07 — Customer Segmentation for an Online Retailer

**Course:** Applied Machine Learning (CSAI2017P), Autumn 2026  
**Theme:** P07 — Retail Segmentation  
**Group:** Team Tesseract

## Project question
Which genuinely distinct customer groups exist in online retail transactions, and can a model assign a new customer to a segment using only their early purchase behaviour?

## Dataset
Use [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Download the dataset ZIP in a browser, extract the workbook, and place it at `data/raw/online_retail_II.xlsx`. The cleaning script reads both workbook sheets. Do not commit the raw ZIP/workbook; document download instructions and source details for reproducibility.

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

The workbook was inspected and the cleaning pipeline was run on both sheets.

| Source sheet | Rows | Columns |
|---|---:|---:|
| Year 2009–2010 | 525,461 | 8 |
| Year 2010–2011 | 541,910 | 8 |
| **Combined** | **1,067,371** | **8** |

The current baseline retains **776,672 transactions (72.765%)** after sequential cleaning. The exact stage-by-stage counts and rules are recorded in [results/cleaning_funnel.csv](results/cleaning_funnel.csv) and [results/cleaning_summary.json](results/cleaning_summary.json).

Cleaning decisions:
- Remove exact duplicate rows.
- Remove transactions without Customer ID because customer-level RFM requires customer identity.
- Exclude cancellation invoices whose Invoice value begins with `C`; these are not netted against original purchases.
- Exclude the explicit non-product/service StockCode list in the cleaning script.
- Remove missing descriptions and non-positive quantities or prices.
- Do not automatically cap or remove extreme positive quantity/price values; inspect them during EDA.

Run the pipeline from the repository root:

```bash
python src/clean_retail_data.py
```

Expected local outputs:
- `data/processed/transactions_clean.csv`
- `results/cleaning_funnel.csv`
- `results/cleaning_summary.json`

The processed transaction CSV is approximately 68.8 MB and remains excluded from Git to keep the repository lightweight. Generate it locally using the command above. The original workbook must also remain outside Git.

## Planned analysis
1. **EDA:** cleaning funnel, monthly revenue, top products and country analysis.
2. **RFM:** customer-level Recency, Frequency and Monetary values with a fixed reference date; inspect skew and scale features.
3. **Clustering:** K-Means for k=2–10 (elbow/inertia and silhouette), hierarchical agglomerative clustering (dendrogram and at least two linkages), and DBSCAN (k-distance plot/noise analysis).
4. **Interpretation:** cluster sizes and mean RFM profiles, business-facing segment labels/recommendations, and PCA 2D projection.
5. **Classification:** Logistic Regression plus at least one tree-based classifier predicting segment from early-purchase behaviour. Avoid leakage from lifetime RFM or target-defining information.
6. **Evaluation:** accuracy, macro-F1, per-class recall, confusion matrix, and `results/metrics.csv`.
7. **Report:** 4–6 page report, reproducible commands and honest limitations.

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
**Clustering and classification metrics pending.** The Phase 1 cleaning retention is a data-preparation statistic, not a model-performance metric.

## AI and external resources
Acknowledge AI tools and external resources used. Verify and understand all retained code; the individual assessment tests understanding.

## Course milestones
- `checkpoint-1`: actual data loaded, shape/head/chart, RFM table, and one baseline K-Means run; create the annotated tag when genuinely complete.
- `checkpoint-2`: full clustering pipeline and initial results; tag only when genuinely complete.
- Final: analysis, evaluation, report, README and `REPRODUCIBILITY.md`.

## Current status
**Phase 1 completed:** both workbook sheets loaded, cleaning rules applied, cleaning funnel and summary recorded. EDA, RFM construction, clustering, classification and final report remain pending.
