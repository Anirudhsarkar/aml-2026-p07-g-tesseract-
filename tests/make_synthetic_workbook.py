"""Create a SYNTHETIC workbook with the Online Retail II schema (for tests only).

The data is random with five built-in customer types; it is NOT the UCI data and
no result computed from it may be reported.
Usage: python tests/make_synthetic_workbook.py OUTPUT.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09 12:50")
TYPES = {  # share, orders per active year, lines per order, price scale, active-window rule
    "champion": (0.08, 14, 14, 3.0, "late"),
    "loyal": (0.20, 6, 9, 2.0, "full"),
    "at_risk": (0.20, 6, 8, 2.0, "early"),
    "new": (0.17, 2, 6, 1.5, "recent"),
    "occasional": (0.35, 1.2, 4, 1.2, "any"),
}


def build(seed=7, n_customers=1500):
    rng = np.random.default_rng(seed)
    span = (END - START).days
    rows, invoice_no = [], 500000
    kinds = rng.choice(list(TYPES), size=n_customers, p=[v[0] for v in TYPES.values()])
    stock = [f"{20000 + i}" for i in range(300)]
    base_price = dict(zip(stock, np.round(rng.lognormal(0.9, 0.6, 300), 2).clip(0.3, 60)))
    for cid, kind in zip(range(12000, 12000 + n_customers), kinds):
        _, rate, lines, price_scale, window = TYPES[kind]
        if window == "late":
            lo, hi = 0, span
        elif window == "full":
            lo, hi = 0, span
        elif window == "early":
            lo = int(rng.integers(0, 200)); hi = lo + int(rng.integers(120, 330))
        elif window == "recent":
            lo = int(rng.integers(span - 120, span - 5)); hi = span
        else:
            lo = int(rng.integers(0, span - 30)); hi = min(span, lo + int(rng.integers(5, 200)))
        n_orders = max(1, int(rng.poisson(rate * (hi - lo) / 365)))
        if window in ("late", "recent"):
            days = np.sort(rng.integers(max(lo, hi - 300), hi + 1, n_orders))
        else:
            days = np.sort(rng.integers(lo, hi + 1, n_orders))
        country = "United Kingdom" if rng.random() < 0.9 else str(rng.choice(["France", "Germany", "EIRE"]))
        for day in days:
            invoice_no += 1
            when = START + pd.Timedelta(days=int(day), hours=int(rng.integers(8, 17)), minutes=int(rng.integers(0, 60)))
            if when > END:
                when = END - pd.Timedelta(minutes=int(rng.integers(1, 600)))
            for code in rng.choice(stock, size=max(1, int(rng.poisson(lines))), replace=False):
                qty = int(rng.integers(1, 12) * (3 if kind == "champion" else 1))
                price = round(float(base_price[code]) * float(rng.uniform(0.9, 1.1)) * price_scale / 2, 2) or 0.5
                rows.append([str(invoice_no), code, f"ITEM {code}", qty, when, price, float(cid), country])
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    # guest orders (no Customer ID), junk codes, invalid prices
    guests = df.sample(frac=0.04, random_state=1).copy(); guests["Customer ID"] = np.nan
    junk = df.sample(frac=0.01, random_state=2).copy(); junk["StockCode"] = rng.choice(["POST", "M", "DOT"], len(junk))
    bad = df.sample(frac=0.002, random_state=3).copy(); bad["Price"] = 0.0
    # cancellations: reverse some purchases later (same customer, code, quantity)
    cancel = df.sample(frac=0.03, random_state=4).copy()
    cancel["Invoice"] = "C" + cancel["Invoice"]; cancel["Quantity"] = -cancel["Quantity"]
    cancel["InvoiceDate"] = cancel["InvoiceDate"] + pd.Timedelta(hours=3)
    # one huge order that is immediately reversed (like the real 80,995-unit order)
    big = pd.DataFrame([["999001", "23843", "PAPER CRAFT , LITTLE BIRDIE", 80995, pd.Timestamp("2011-12-09 09:15"), 2.08, 12001.0, "United Kingdom"],
                        ["C999002", "23843", "PAPER CRAFT , LITTLE BIRDIE", -80995, pd.Timestamp("2011-12-09 09:27"), 2.08, 12001.0, "United Kingdom"]],
                       columns=df.columns)
    out = pd.concat([df, guests, junk, bad, cancel, big, df.sample(frac=0.01, random_state=5)], ignore_index=True)
    out["InvoiceDate"] = out["InvoiceDate"].clip(upper=END)
    return out.sort_values("InvoiceDate").reset_index(drop=True)


def main():
    target = Path(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    cut = pd.Timestamp("2010-12-01")
    with pd.ExcelWriter(target) as writer:
        data[data.InvoiceDate < cut].to_excel(writer, sheet_name="Year 2009-2010", index=False)
        data[data.InvoiceDate >= cut].to_excel(writer, sheet_name="Year 2010-2011", index=False)
    print(f"Wrote synthetic workbook with {len(data):,} rows to {target}")


if __name__ == "__main__":
    main()
