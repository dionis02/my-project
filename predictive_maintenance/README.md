# Predictive maintenance sklearn solution

This folder contains a reproducible sklearn workflow for the industrial equipment failure competition.

## Files

- `solution.py` — reusable experiment runner and final submission pipeline.
- `experiments/` — target directory for grid-search CSV files, validation predictions, trained models, and figures.
- `final_solution.ipynb` — notebook that imports `solution.py`, runs the same workflow, and writes `fin_submission.csv`.

## Recommended run

```bash
python predictive_maintenance/solution.py --quick
```

After validating the environment, run the full grid search:

```bash
python predictive_maintenance/solution.py
```

By default the script uses `--n-jobs 1` to avoid sklearn/joblib parallel warnings such as
`sklearn.utils.parallel.delayed should be used with sklearn.utils.parallel.Parallel`.
If your local environment handles sklearn parallelism cleanly, you can opt in to parallel execution:

```bash
python predictive_maintenance/solution.py --n-jobs -1
```

The pipeline compares multiple sklearn model families, multiple imputation approaches, sensor-only features versus available failure-mode flags, and domain features including:

```text
Power = 2 * pi * Torque * Rotational_speed / 60
```
