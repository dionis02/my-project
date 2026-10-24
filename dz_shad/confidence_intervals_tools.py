from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

ArrayLike = Union[np.ndarray, list]
NumericArray = Union[List[float], np.ndarray]

def plot_point_estimates_vs_truth(
    point_estimates: np.ndarray,
    true_parameter: float = 0.0,
    margin: float = 0.2,
    figsize: tuple = (7, 2.2),
) -> None:
    """
    Визуализирует точечные оценки параметра в сравнении с его истинным значением.

    Args:
        point_estimates: Массив точечных оценок, полученных из симуляций.
        true_parameter: Истинное значение параметра, которое мы пытаемся оценить.
        margin: Относительный отступ по краям графика для оси X.
        figsize: Размер фигуры matplotlib (ширина, высота в дюймах).
    """
    point_estimates = np.asarray(point_estimates)
    num_estimates = len(point_estimates)
    y_levels = np.arange(num_estimates)[::-1]

    # Определяем границы для оси X
    min_val = min(point_estimates.min(), true_parameter)
    max_val = max(point_estimates.max(), true_parameter)
    padding = (max_val - min_val) * margin if max_val > min_val else margin
    xlim = (min_val - padding, max_val + padding)

    # График
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 0.25 * num_estimates))
    ax.set_yticks([])
    ax.set_xlabel("Значения параметра")
    ax.set_xlim(*xlim)

    # Вертикальная линия для истинного значения
    ax.vlines(true_parameter, -1, num_estimates, color="tab:blue", lw=3, alpha=0.8, label="Истина")

    # Отображаем каждую оценку и ее отклонение от истины
    for i, (y_pos, estimate) in enumerate(zip(y_levels, point_estimates)):
        ax.hlines(y_pos, *sorted([estimate, true_parameter]), color="purple", lw=2.5, alpha=0.85)
        ax.plot(estimate, y_pos, "o", color="tab:red", ms=8, label="Оценка" if i == 0 else "")
        ax.plot(true_parameter, y_pos, "o", color="tab:blue", ms=8)

    ax.legend(loc="upper left")
    ax.set_title("Отклонения точечных оценок параметра")
    plt.tight_layout()
    plt.show()

def plot_regression_with_residuals(
    feature: np.ndarray,
    target: np.ndarray,
    xlabel: str = "Признак X",
    ylabel: str = "Таргет Y",
    point_color: str = "tab:red",
    resid_color: str = "purple",
) -> None:
    """
    Строит график линейной регрессии и визуализирует остатки.

    Args:
        feature: Вектор признаков для оси X.
        target: Вектор целевой переменной для оси Y.
        xlabel: Подпись для оси X.
        ylabel: Подпись для оси Y.
        point_color: Цвет точек данных (наблюдений).
        resid_color: Цвет линий, отображающих остатки.
    """
    # Линейная регрессия
    feature = feature.reshape(-1, 1)
    model = LinearRegression().fit(feature, target)
    predicted_values = model.predict(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(feature, target, s=36, color=point_color, zorder=3, label="Наблюдения")
    x_line = np.linspace(feature.min(), feature.max(), 300).reshape(-1, 1)
    ax.plot(x_line, model.predict(x_line), color="tab:blue", lw=2.5, label="Линия регрессии")

    # Рисуем остатки
    for x_val, y_val, y_pred in zip(feature, target, predicted_values):
        ax.vlines(x_val, y_val, y_pred, color=resid_color, lw=2, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Остатки относительно линии регрессии")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_parameter_interval(
    theta_hat: float, delta: float, theta_true: float = None, figsize: tuple = (8, 1.5)
) -> None:
    """
    Визуализирует точечную оценку, интервал и истинное значение параметра.

    Args:
        theta_true: Истинное значение параметра (синяя точка).
        theta_hat: Точечная оценка параметра (центр интервала).
        delta: Полуширина интервала (расстояние от оценки до края).
        figsize: Размер фигуры matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        x=theta_hat,
        y=0,
        xerr=delta,
        fmt="*",
        color="tab:red",
        ecolor="mediumseagreen",
        lw=6,
        markersize=20,
        capsize=15,
        capthick=5,
        label=r"Оценка $\hat{\theta}$ и её интервал",
        zorder=3,
    )

    if theta_true:
        ax.scatter(
            theta_true, 0, s=100, color="tab:blue", label=r"Истинное значение $\theta$", zorder=4
        )

    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=False)
    plt.show()

def plot_intervals(
    num_experiments: int,
    lower_bounds: NumericArray,
    upper_bounds: NumericArray,
    sample_means: NumericArray,
    target: Union[float, NumericArray],
    title: str,
) -> None:
    """
    Строит график доверительных или предсказательных интервалов.

    Args:
        num_experiments (int): Общее количество проведенных экспериментов.
        lower_bounds (NumericArray): Нижние границы интервалов для каждого эксперимента.
        upper_bounds (NumericArray): Верхние границы интервалов для каждого эксперимента.
        sample_means (NumericArray): Выборочные средние для каждого эксперимента.
        target (Union[float, NumericArray]): Истинное значение параметра или таргета.
        title (str): Заголовок, который будет отображен на графике.
    """
    # покрытие интервалом цели
    is_covered = (np.array(lower_bounds) <= target) & (target <= np.array(upper_bounds))

    fig, ax = plt.subplots(figsize=(10, 10))
    legend_elements = [
        Line2D([0], [0], color="green", lw=3, label="Интервал покрывает цель"),
        Line2D([0], [0], color="red", lw=3, label="Интервал не покрывает цель"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linestyle="None",
            markersize=5,
            label="Выборочное среднее",
        ),
    ]

    # Способы отрисобки таргета
    if isinstance(target, (int, float)):
        # Если параметр один
        ax.axvline(target, color="blue", alpha=0.5, linestyle="--")
        legend_elements.append(
            Line2D(
                [0], [0], color="blue", lw=2, linestyle="--", label=f"Истинное значение = {target}"
            )
        )
    else:
        # Если это массив
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="b",
                linestyle="None",
                markersize=7,
                label="Элемент тестовой выборки",
            )
        )

    for i in range(num_experiments):
        y_pos = i + 1
        color = "green" if is_covered[i] else "red"

        # Рисуем интервалы и среднее
        ax.plot([lower_bounds[i], upper_bounds[i]], [y_pos, y_pos], color=color, linewidth=2.3)
        ax.plot(sample_means[i], y_pos, "ko", markersize=3)

        # Если target - массив, рисуем его элементы
        if not isinstance(target, (int, float)):
            ax.plot(target[i], y_pos, "b*", markersize=5)

    # Настройки осей и заголовка
    ax.set_ylim(0, num_experiments + 1)
    ax.set_xlabel(r"Значение параметра $\theta$")
    ax.set_ylabel("Номер эксперимента")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_feature_intervals(
    intervals: pd.DataFrame,
    features_to_show: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    pad: float = 0.35,
) -> None:
    """
    Визуализирует доверительные интервалы средних признаков по классам.

    Args:
        intervals: DataFrame с колонками ['feature', 'class', 'mean', 'left', 'right'].
        features_to_show: Список признаков для отображения. Если None, покажет первые 5.
        colors: Палитра цветов для классов. Если None, используется палитра Seaborn.
        pad: Вертикальный отступ для интервалов на каждом подграфике.
    """

    if features_to_show is None:
        features_to_show = intervals["feature"].unique()[:5]

    classes = sorted(intervals["class"].unique())
    if colors is None:
        colors = sns.color_palette("Set1", n_colors=len(classes))
    class_to_y_pos = {cls: i for i, cls in enumerate(classes)}
    class_to_color = dict(zip(classes, colors))

    fig, axes = plt.subplots(
        len(features_to_show), 1, figsize=(8, 2 * len(features_to_show)), sharex=False
    )
    if len(features_to_show) == 1:
        axes = [axes]

    for ax, feature_name in zip(axes, features_to_show):
        feature_data = intervals[intervals["feature"] == feature_name]

        for _, row in feature_data.iterrows():
            class_label = row["class"]
            y_pos = class_to_y_pos[class_label]
            с = class_to_color[class_label]

            ax.hlines(y_pos, row["left"], row["right"], color=с, lw=5, alpha=0.8)
            ax.scatter(row["mean"], y_pos, s=80, color=с, zorder=3)

        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([f"класс {c}" for c in classes])
        ax.set_ylim(-pad, len(classes) - 1 + pad)
        ax.set_title(feature_name)

    plt.tight_layout()
    plt.show()

def draw_confidence_interval(
    left: ArrayLike,
    right: ArrayLike,
    estimation: Optional[ArrayLike] = None,
    sample: Optional[ArrayLike] = None,
    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
    color_estimation: str = "#FF3300",
    color_interval: str = "#00CC66",
    color_sample: str = "#0066FF",
    label_estimation: Optional[str] = None,
    label_interval: Optional[str] = None,
) -> None:
    """Рисует доверительный интервал и оценку в зависимости от размера выборки.

    Args:
        left (ArrayLike): Массив левых (нижних) границ интервалов.
        right (ArrayLike): Массив правых (верхних) границ интервалов.
        estimation (Optional[ArrayLike]): Массив точечных оценок (по умолчанию None).
        sample (Optional[ArrayLike]): Исходная выборка для отображения (по умолчанию None).
        ylim (tuple, optional): Кортеж для ограничения вертикальной оси (ymin, ymax).
        color_estimation (str): Цвет для линии точечной оценки.
        color_interval (str): Цвет для закрашенной области интервала.
        color_sample (str): Цвет для точек исходной выборки.
        label_estimation (Optional[str]): Текстовая метка для линии оценки в легенде.
        label_interval (Optional[str]): Текстовая метка для области интервала в легенде.
    """

    sample_sizes = np.arange(len(left)) + 1

    assert len(sample_sizes) == len(right)

    if not sample is None:
        plt.scatter(
            sample_sizes,
            sample,
            alpha=0.3,
            color=color_sample,
            s=40,
            label="Элемент выборки",
        )

    if not estimation is None:
        plt.plot(
            sample_sizes,
            estimation,
            color=color_estimation,
            linewidth=2.5,
            label=label_estimation,
        )

    plt.fill_between(
        sample_sizes,
        left,
        right,
        alpha=0.25,
        color=color_interval,
        label=label_interval,
    )
    plt.legend()
    plt.ylim(ylim)
    plt.xlabel("Размер выборки")
    plt.title("Зависимость доверительного интервала от размера выборки")
