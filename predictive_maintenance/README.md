# Predictive maintenance sklearn solution

This folder contains a reproducible sklearn workflow for the industrial equipment failure competition.

## Files

- `solution.py` — reusable experiment runner and final submission pipeline.
- `experiments/` — target directory for grid-search CSV files, validation predictions, trained models, and figures.
- `final_solution.ipynb` — self-contained notebook version of the workflow. It does not import helper code from local `.py` files and can reproduce `fin_submission.csv` (plus the compatibility alias `for_submission.csv`) from `train.csv`, `test.csv`, and `sample_submission.csv`.

## Recommended run

First validate that the required packages are available:

```bash
python -m pip install -r predictive_maintenance/requirements.txt --dry-run
```

Run a fast smoke test:

```bash
python predictive_maintenance/solution.py --quick
```

Run the compact model-selection experiment used for the submitted artifact:

```bash
python predictive_maintenance/solution.py
```

The pipeline compares multiple sklearn model families and compact hyperparameter settings. The five three-letter failure-mode flag columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from the feature matrix, along with aggregates derived from them. It keeps sensor and domain-engineered features including:

```text
Power = 2 * pi * Torque * Rotational_speed / 60
```
