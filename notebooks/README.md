# Notebook run order

Create the required analysis notebooks in this order:
1. `01-eda.ipynb`
2. `02-preprocessing.ipynb`
3. `03-models.ipynb`
4. `04-evaluation.ipynb`

The reusable cleaning pipeline must remain under `src/`; notebooks should consume its outputs rather than repeat one-off cleaning logic.
