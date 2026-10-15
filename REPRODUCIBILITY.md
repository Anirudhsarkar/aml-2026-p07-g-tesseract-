# Reproducibility

## Environment
- Python version: TODO (use the same version across the team; Python 3.11/3.12 recommended)
- OS and hardware: TODO
- Install dependencies: `python -m pip install -r requirements.txt`

## Dataset
1. Download UCI Online Retail II from https://archive.ics.uci.edu/dataset/502/online+retail+ii using a browser.
2. Extract the workbook to `data/raw/online_retail_II.xlsx`.
3. Do not disable TLS/SSL verification to work around download problems.

## Run sequence
1. `python src/clean_retail_data.py`
2. Run notebooks in the documented order under `notebooks/`.
3. TODO: add final commands, expected outputs, runtime and hardware after the pipeline is implemented.

## Randomness
Record every random seed and where it is set: TODO.

## Reproducibility limitations
- The cleaning script is a preliminary baseline and has not yet been run on the real workbook.
- Cancellation invoices are excluded rather than matched/netted against original invoices; review and document this decision.
- Extreme positive quantities are not automatically removed or capped; inspect outliers before modelling.
- Record any further manual steps and known nondeterminism here.
