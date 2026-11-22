# Dataset location

The project uses the UCI Online Retail II dataset:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Download the ZIP from UCI in a browser and extract online_retail_II.xlsx.

Do not commit the raw workbook or ZIP to this repository. The workbook is approximately 43.5 MB and should remain local, outside Git history. Place it at data/raw/online_retail_II.xlsx.

Then run:

    python src/clean_retail_data.py
    python src/run_eda.py

The data/raw/ and data/processed/ folders are excluded from Git. This file documents the dataset source and placement instructions; it is not a copy of the dataset.
