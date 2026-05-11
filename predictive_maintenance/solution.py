"""Reproducible sklearn solution for the predictive-maintenance competition.

The module is intentionally notebook-friendly: the final notebook imports the
functions below, runs the same grid-search experiments, saves diagnostics into a
single experiment directory, trains the selected model on all labelled data, and
writes ``fin_submission.csv`` with probabilities from ``predict_proba``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
ID_COLUMN = "id"
DROP_COLUMNS = ["Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the experiment runner and final notebook."""

    train_path: Path = Path("train.csv")
    test_path: Path = Path("test.csv")
    sample_submission_path: Path = Path("sample_submission.csv")
    output_dir: Path = Path("predictive_maintenance/experiments")
    cv_splits: int = 5
    quick: bool = False
    n_jobs: int = 1


def read_data(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample-submission files."""

    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)
    sample_submission = pd.read_csv(config.sample_submission_path)
    return train, test, sample_submission


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired features without mutating the original dataframe."""

    out = df.copy()
    air = out["Air temperature [K]"]
    process = out["Process temperature [K]"]
    speed = out["Rotational speed [rpm]"]
    torque = out["Torque [Nm]"]
    wear = out["Tool wear [min]"]

    out["temperature_delta"] = process - air
    out["temperature_ratio"] = process / air.replace(0, np.nan)
    out["mechanical_power"] = 2 * np.pi * torque * speed / 60
    out["torque_x_wear"] = torque * wear
    out["speed_x_torque"] = speed * torque
    out["wear_per_power"] = wear / out["mechanical_power"].replace(0, np.nan)
    out["torque_per_speed"] = torque / speed.replace(0, np.nan)

    if set(FAILURE_MODE_COLUMNS).issubset(out.columns):
        out["failure_mode_sum"] = out[FAILURE_MODE_COLUMNS].sum(axis=1)
        out["any_failure_mode_flag"] = (out["failure_mode_sum"] > 0).astype(int)

    return out


def split_features_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_failure_mode_flags: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create aligned train/test feature matrices and target vector."""

    train_fe = add_engineered_features(train)
    test_fe = add_engineered_features(test)
    y = train_fe[TARGET].astype(int)

    drop_cols = [TARGET, ID_COLUMN, *DROP_COLUMNS]
    if not include_failure_mode_flags:
        drop_cols.extend([*FAILURE_MODE_COLUMNS, "failure_mode_sum", "any_failure_mode_flag"])

    X = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
    X_test = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns])
    X_test = X_test.reindex(columns=X.columns)
    return X, y, X_test


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing with tunable imputers for numeric and categorical columns."""

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = [("num", Pipeline(numeric_steps), numeric_features)]
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def model_search_spaces(quick: bool = False, n_jobs: int = 1) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return sklearn models and GridSearchCV parameter grids."""

    logistic = Pipeline(
        steps=[
            ("preprocess", "passthrough"),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    hgb = Pipeline(
        steps=[
            ("preprocess", "passthrough"),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True)),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("preprocess", "passthrough"),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, class_weight="balanced_subsample")),
        ]
    )
    extra_trees = Pipeline(
        steps=[
            ("preprocess", "passthrough"),
            ("model", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, class_weight="balanced")),
        ]
    )

    if quick:
        return {
            "logistic_regression": (
                logistic,
                {
                    "preprocess__num__imputer__strategy": ["median"],
                    "preprocess__cat__imputer__strategy": ["most_frequent"],
                    "model__C": [1.0],
                },
            ),
            "hist_gradient_boosting": (
                hgb,
                {
                    "preprocess__num__imputer__strategy": ["median"],
                    "preprocess__cat__imputer__strategy": ["most_frequent"],
                    "model__learning_rate": [0.06],
                    "model__max_leaf_nodes": [31],
                    "model__max_iter": [250],
                    "model__l2_regularization": [0.0],
                },
            ),
        }

    return {
        "logistic_regression": (
            logistic,
            {
                "preprocess__num__imputer__strategy": ["median", "mean", "constant"],
                "preprocess__num__imputer__fill_value": [0],
                "preprocess__cat__imputer__strategy": ["most_frequent", "constant"],
                "preprocess__cat__imputer__fill_value": ["missing"],
                "model__C": [0.05, 0.1, 1.0, 5.0, 10.0],
            },
        ),
        "hist_gradient_boosting": (
            hgb,
            {
                "preprocess__num__imputer__strategy": ["median", "mean"],
                "preprocess__cat__imputer__strategy": ["most_frequent", "constant"],
                "preprocess__cat__imputer__fill_value": ["missing"],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__max_iter": [250, 450],
                "model__l2_regularization": [0.0, 0.05, 0.2],
            },
        ),
        "random_forest": (
            random_forest,
            {
                "preprocess__num__imputer__strategy": ["median", "mean"],
                "preprocess__cat__imputer__strategy": ["most_frequent", "constant"],
                "preprocess__cat__imputer__fill_value": ["missing"],
                "model__n_estimators": [300, 600],
                "model__max_depth": [8, 14, None],
                "model__min_samples_leaf": [1, 5, 15],
                "model__max_features": ["sqrt", 0.8],
            },
        ),
        "extra_trees": (
            extra_trees,
            {
                "preprocess__num__imputer__strategy": ["median", "mean"],
                "preprocess__cat__imputer__strategy": ["most_frequent", "constant"],
                "preprocess__cat__imputer__fill_value": ["missing"],
                "model__n_estimators": [400, 700],
                "model__max_depth": [8, 14, None],
                "model__min_samples_leaf": [1, 5, 15],
                "model__max_features": ["sqrt", 0.8],
            },
        ),
    }


def save_eda_plots(train: pd.DataFrame, output_dir: Path) -> None:
    """Save basic EDA charts into the experiment folder."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    train_fe = add_engineered_features(train)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_fe, x=TARGET)
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=160)
    plt.close()

    missing = train_fe.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing fraction")
    plt.title("Missing values by feature")
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_values.png", dpi=160)
    plt.close()

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temperature_delta",
        "mechanical_power",
    ]
    axes = train_fe[numeric_cols].hist(figsize=(12, 8), bins=40)
    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_feature_histograms.png", dpi=160)
    plt.close()

    corr_cols = [c for c in train_fe.select_dtypes(include=["number", "bool"]).columns if c != ID_COLUMN]
    corr = train_fe[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def run_grid_searches(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cv_splits: int,
    quick: bool,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, Pipeline]:
    """Run stratified ROC-AUC GridSearchCV for several sklearn model families."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[pd.DataFrame] = []
    best_estimators: dict[str, Pipeline] = {}

    for model_name, (pipe, grid) in model_search_spaces(quick=quick, n_jobs=n_jobs).items():
        scale_numeric = model_name == "logistic_regression"
        pipe.set_params(preprocess=make_preprocessor(X, scale_numeric=scale_numeric))
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=1,
            return_train_score=True,
        )
        search.fit(X, y)
        result = pd.DataFrame(search.cv_results_)
        result.insert(0, "model_name", model_name)
        result.to_csv(output_dir / f"grid_results_{model_name}.csv", index=False)
        rows.append(result)
        best_estimators[model_name] = search.best_estimator_
        joblib.dump(search.best_estimator_, output_dir / f"best_{model_name}.joblib")

    all_results = pd.concat(rows, ignore_index=True)
    # ``rank_test_score`` is local to each GridSearchCV run. After concatenating
    # multiple model families, choose the global winner by cross-validated ROC AUC
    # instead of by per-family ranks, otherwise every family contributes a rank-1
    # row and insertion order can incorrectly decide the selected estimator.
    all_results = all_results.sort_values(
        ["mean_test_score", "std_test_score"],
        ascending=[False, True],
    ).reset_index(drop=True)
    all_results.insert(1, "overall_rank_test_score", np.arange(1, len(all_results) + 1))
    all_results.to_csv(output_dir / "grid_results_all.csv", index=False)
    best_model_name = all_results.iloc[0]["model_name"]
    return all_results, best_estimators[str(best_model_name)]


def save_validation_plots(
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cv_splits: int,
    n_jobs: int = 1,
) -> float:
    """Save an out-of-fold ROC curve for the selected model."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    auc = roc_auc_score(y, oof_proba)

    RocCurveDisplay.from_predictions(y, oof_proba)
    plt.title(f"Out-of-fold ROC curve, AUC={auc:.5f}")
    plt.tight_layout()
    plt.savefig(figures_dir / "best_model_oof_roc.png", dpi=160)
    plt.close()

    pd.DataFrame({ID_COLUMN: X.index, "oof_probability": oof_proba, TARGET: y}).to_csv(
        output_dir / "best_model_oof_predictions.csv", index=False
    )
    return float(auc)


def train_and_write_submission(
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path = Path("fin_submission.csv"),
) -> pd.DataFrame:
    """Fit the selected estimator on all training data and write final probabilities."""

    best_model.fit(X, y)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    submission = sample_submission.copy()
    submission[TARGET] = probabilities
    submission.to_csv(output_path, index=False)
    return submission


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """End-to-end experiment entry point used by both CLI and notebook."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample_submission = read_data(config)
    save_eda_plots(train, config.output_dir)

    summaries = []
    best_overall_auc = -np.inf
    best_overall_model: Pipeline | None = None
    best_overall_features: tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None = None
    best_flag_setting = True

    for include_flags in [False, True]:
        setting_dir = config.output_dir / ("with_failure_flags" if include_flags else "sensor_features_only")
        X, y, X_test = split_features_target(train, test, include_failure_mode_flags=include_flags)
        results, best_model = run_grid_searches(X, y, setting_dir, config.cv_splits, config.quick, config.n_jobs)
        auc = save_validation_plots(best_model, X, y, setting_dir, config.cv_splits, config.n_jobs)
        best_row = results.iloc[0].to_dict()
        best_row["include_failure_mode_flags"] = include_flags
        best_row["best_oof_auc"] = auc
        summaries.append(best_row)
        if auc > best_overall_auc:
            best_overall_auc = auc
            best_overall_model = best_model
            best_overall_features = (X, y, X_test)
            best_flag_setting = include_flags

    summary = pd.DataFrame(summaries).sort_values("best_oof_auc", ascending=False)
    summary.to_csv(config.output_dir / "experiment_summary.csv", index=False)
    (config.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2, default=str), encoding="utf-8"
    )

    if best_overall_model is None or best_overall_features is None:
        raise RuntimeError("No model was fitted. Check the experiment configuration.")

    X, y, X_test = best_overall_features
    submission = train_and_write_submission(
        best_overall_model,
        X,
        y,
        X_test,
        sample_submission,
        output_path=Path("fin_submission.csv"),
    )
    joblib.dump(best_overall_model, config.output_dir / "final_model.joblib")
    return {
        "best_oof_auc": best_overall_auc,
        "include_failure_mode_flags": best_flag_setting,
        "submission_rows": len(submission),
        "output_dir": str(config.output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sklearn experiments and create fin_submission.csv.")
    parser.add_argument("--train-path", type=Path, default=Path("train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("test.csv"))
    parser.add_argument("--sample-submission-path", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictive_maintenance/experiments"))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid before the full search.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel jobs for sklearn searches and tree estimators. "
            "The default keeps execution serial to avoid sklearn/joblib parallel warnings "
            "and nested parallelism; set -1 if your local sklearn/joblib stack supports it."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_submission_path=args.sample_submission_path,
        output_dir=args.output_dir,
        cv_splits=args.cv_splits,
        quick=args.quick,
        n_jobs=args.n_jobs,
    )
    result = run_experiment(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
