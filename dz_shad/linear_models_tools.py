from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_regression_intervals(
    train_data: pd.DataFrame,
    pred_summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    formula_text: Optional[str] = None,
    pred_interval_cols: Tuple[str, str] = ("obs_ci_lower", "obs_ci_upper"),
    pred_interval_label: str = "95% Предикт. инт.",
) -> None:
    """
    Строит график регрессии с доверительным и предиктивным интервалами.

    Args:
        train_data (pd.DataFrame): DataFrame с исходными данными для scatter plot.
        pred_summary (pd.DataFrame): DataFrame с предсказаниями, полученный из
            `.summary_frame()`. Должен содержать x_col, 'mean', 'mean_ci_lower',
            'mean_ci_upper' и колонки, указанные в pred_interval_cols.
        x_col (str): Название колонки с независимой переменной (ось X).
        y_col (str): Название колонки с зависимой переменной (ось Y).
        title (str): Заголовок графика.
        formula_text (Optional[str], optional): Текст с формулой модели для
            отображения на графике. По умолчанию None (не отображать).
        pred_interval_cols (Tuple[str, str], optional): Кортеж из двух строк с
            названиями колонок для нижней и верхней границ предиктивного
            интервала. По умолчанию ('obs_ci_lower', 'obs_ci_upper').
        pred_interval_label (str, optional): Метка для легенды предиктивного
            интервала. По умолчанию '95% Предикт. инт.'.
    """
    plt.figure(figsize=(10, 6))

    # Исходные данные
    sns.scatterplot(
        x=x_col, y=y_col, data=train_data, alpha=0.5, color="#20b2aa", label="Обучающая выборка"
    )

    # Линия регрессии
    plt.plot(
        pred_summary[x_col],
        pred_summary["mean"],
        color="#ff2400",
        linewidth=2,
        label="Предсказания",
    )

    # Доверительный интервал
    plt.fill_between(
        pred_summary[x_col],
        pred_summary["mean_ci_lower"],
        pred_summary["mean_ci_upper"],
        color="#8b00ff",
        alpha=0.7,
        label="95% Доверит. инт.",
    )

    # Предиктивный интервал
    lower_col, upper_col = pred_interval_cols
    plt.fill_between(
        pred_summary[x_col],
        pred_summary[lower_col],
        pred_summary[upper_col],
        color="#8b00ff",
        alpha=0.2,
        label=pred_interval_label,
    )

    plt.title(title)
    plt.xlabel(f"Общая площадь ({x_col})")
    plt.ylabel(f"Цена ({y_col})")

    if formula_text:
        plt.text(
            0.05,
            0.95,
            formula_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="grey", alpha=0.2),
        )

    plt.ylim(-50, None)
    plt.legend(loc=4, fontsize=11)
    plt.show()