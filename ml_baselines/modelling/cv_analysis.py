import math
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ml_baselines.config import Config
cfg = Config()

def load_cv_results(site, save_suffix, model_type, save_path=cfg.models_path) -> pd.DataFrame:
    """
    Load a CV results CSV saved by train_baseline_model_grid_search.

    Args:
        site: Site name (e.g. "MHD").
        save_suffix: Filename suffix used when saving (None for default name).
        model_type: Model type string (e.g. "gradient_boosting").
        save_path: Root models directory. Defaults to cfg.models_path.

    Returns:
        DataFrame with all columns as-is, plus recalculated rank_test_<metric>
        columns that are global across all data_kwarg groups.
    """
    filename = f"cv_results_{site}_{model_type}_model.csv" if save_suffix is None else f"cv_results_{site}_{model_type}_{save_suffix}.csv"
    filepath = Path(save_path) / site / filename

    if not filepath.exists():
        raise FileNotFoundError(f"CV results file not found: {filepath}")

    df = pd.read_csv(filepath)
    df = _rerank(df)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rerank(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate rank_test_<metric> globally across all data_kwarg groups."""
    df = df.copy()
    for metric in _detect_metrics(df):
        score_col = f"mean_test_{metric}"
        df[f"rank_test_{metric}"] = df[score_col].rank(ascending=False, method="min").astype(int)
    return df


def _detect_metrics(cv_df: pd.DataFrame) -> list[str]:
    """Return all metric names found as mean_test_<metric> columns."""
    return [
        re.fullmatch(r"mean_test_(.+)", c).group(1)
        for c in cv_df.columns
        if re.fullmatch(r"mean_test_(.+)", c)
    ]


def _resolve_metric(cv_df: pd.DataFrame, metric: str | None) -> str:
    """Resolve None to 'f1' if present, otherwise the first available metric."""
    if metric is not None:
        return metric
    available = _detect_metrics(cv_df)
    if not available:
        raise ValueError("No mean_test_<metric> columns found in cv_df.")
    return "f1" if "f1" in available else available[0]


def _param_cols(cv_df: pd.DataFrame) -> list[str]:
    return [c for c in cv_df.columns if c.startswith("param_")]


def _score_display_name(metric: str) -> str:
    return metric.replace("_", " ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_best_params(cv_df: pd.DataFrame, n: int = 1, metric: str | None = None) -> pd.DataFrame:
    """
    Return the top n parameter sets by test score.

    Args:
        cv_df: CV results DataFrame from load_cv_results.
        n: Number of top parameter sets to return.
        metric: Metric to rank by (e.g. "f1", "precision", "recall").
            Defaults to "f1" if present, otherwise the first available metric.

    Returns:
        DataFrame with param_* columns plus test score, train score, and rank,
        sorted best-first.
    """
    metric = _resolve_metric(cv_df, metric)
    score_col = f"mean_test_{metric}"
    keep_cols = _param_cols(cv_df) + [score_col, f"mean_train_{metric}", f"rank_test_{metric}"]
    keep_cols = [c for c in keep_cols if c in cv_df.columns]
    return cv_df[keep_cols].sort_values(score_col, ascending=False).head(n).reset_index(drop=True)


def get_top_params(
    cv_df: pd.DataFrame,
    threshold: float | None = None,
    n: int = 5,
    metric: str | None = None,
) -> pd.DataFrame:
    """
    Return parameter sets that meet a score threshold, or the top n if no
    threshold is given.

    Args:
        cv_df: CV results DataFrame from load_cv_results.
        threshold: Minimum score value. If provided, returns all rows that meet
            or exceed this value, ignoring n.
        n: Number of top rows to return when no threshold is given.
        metric: Metric to filter/sort by. Defaults to "f1" if present,
            otherwise the first available metric.

    Returns:
        Filtered and sorted DataFrame (best-first) with param_* columns,
        test score, train score, and rank columns.
    """
    metric = _resolve_metric(cv_df, metric)
    score_col = f"mean_test_{metric}"
    keep_cols = _param_cols(cv_df) + [score_col, f"mean_train_{metric}", f"rank_test_{metric}"]
    keep_cols = [c for c in keep_cols if c in cv_df.columns]
    df = cv_df[keep_cols].sort_values(score_col, ascending=False)
    if threshold is not None:
        return df[df[score_col] >= threshold].reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def summarise_param_effects(
    cv_df: pd.DataFrame, metric: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    For each param_* column, compute mean, std, and count of the test score
    grouped by that parameter's values.

    Args:
        cv_df: CV results DataFrame from load_cv_results.
        metric: Metric to summarise. Defaults to "f1" if present, otherwise
            the first available metric.

    Returns:
        Dict mapping each param column name to a DataFrame with columns:
        [param_value, mean_score, std_score, count].
    """
    metric = _resolve_metric(cv_df, metric)
    score_col = f"mean_test_{metric}"
    results = {}
    for col in _param_cols(cv_df):
        grouped = (
            cv_df.assign(**{col: cv_df[col].astype(str)})
            .groupby(col, dropna=False)[score_col]
            .agg(mean_score="mean", std_score="std", count="count")
            .reset_index()
        )
        grouped = grouped.rename(columns={col: "param_value"})
        grouped = grouped.sort_values("mean_score", ascending=False).reset_index(drop=True)
        results[col] = grouped
    return results


def plot_param_effects(
    cv_df: pd.DataFrame,
    metric: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """
    Plot a grid of bar charts showing mean score (± std) for each value of
    every param_* column.

    Args:
        cv_df: CV results DataFrame from load_cv_results.
        metric: Metric to plot. Defaults to "f1" if present, otherwise the
            first available metric. Pass "all" to plot all available metrics
            as grouped bars within each subplot.
        figsize: Optional (width, height) tuple. Auto-sized if None.

    Returns:
        (fig, axes) — call plt.show() or fig.savefig() in the caller.
    """
    if metric == "all":
        return _plot_param_effects_all_metrics(cv_df, figsize=figsize)

    metric = _resolve_metric(cv_df, metric)
    effects = summarise_param_effects(cv_df, metric=metric)
    n_params = len(effects)
    if n_params == 0:
        raise ValueError("No param_* columns found in cv_df.")

    ncols = min(3, n_params)
    nrows = math.ceil(n_params / ncols)

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    score_label = _score_display_name(metric)

    for i, (col, grouped) in enumerate(effects.items()):
        ax = axes_flat[i]
        x = np.arange(len(grouped))
        yerr = grouped["std_score"].fillna(0)
        ax.bar(x, grouped["mean_score"], yerr=yerr, capsize=4, color="royalblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["param_value"].astype(str), rotation=30, ha="right")
        ax.set_title(col.replace("param_", ""))
        ax.set_ylabel(score_label)
        ax.set_ylim(0, 1)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Mean {score_label} by parameter value", fontsize=13)
    fig.tight_layout()
    return fig, axes


def _plot_param_effects_all_metrics(cv_df: pd.DataFrame, figsize: tuple | None = None) -> tuple:
    """Grouped-bar variant of plot_param_effects with one bar group per metric."""
    metrics = _detect_metrics(cv_df)
    if not metrics:
        raise ValueError("No mean_test_<metric> columns found in cv_df.")

    # Build {metric: {param_col: grouped_df}} by reusing summarise_param_effects
    effects_by_metric = {m: summarise_param_effects(cv_df, metric=m) for m in metrics}

    param_cols = list(next(iter(effects_by_metric.values())).keys())
    n_params = len(param_cols)
    if n_params == 0:
        raise ValueError("No param_* columns found in cv_df.")

    ncols = min(3, n_params)
    nrows = math.ceil(n_params / ncols)

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, col in enumerate(param_cols):
        ax = axes_flat[i]
        # Use param_value order from first metric's grouped df (sorted by mean score)
        param_values = effects_by_metric[metrics[0]][col]["param_value"].tolist()
        x = np.arange(len(param_values))

        for m_idx, m in enumerate(metrics):
            grouped = effects_by_metric[m][col].set_index("param_value").reindex(param_values)
            offset = (m_idx - (n_metrics - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                grouped["mean_score"],
                yerr=grouped["std_score"].fillna(0),
                width=bar_width,
                capsize=3,
                color=colors[m_idx % len(colors)],
                alpha=0.8,
                label=m,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(param_values, rotation=30, ha="right")
        ax.set_title(col.replace("param_", ""))
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        for level in [0.2, 0.4, 0.6, 0.8]:
            ax.axhline(level, color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Mean scores by parameter value", fontsize=13)
    fig.tight_layout()
    return fig, axes


def plot_score_heatmap(
    cv_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str | None = None,
    ax=None,
) -> tuple:
    """
    Plot a 2D heatmap of mean score across two parameters.

    Args:
        cv_df: CV results DataFrame from load_cv_results.
        param_x: Parameter for the x-axis. Can be given with or without the
            'param_' prefix (e.g. 'max_depth' or 'param_max_depth').
        param_y: Parameter for the y-axis.
        metric: Metric to aggregate. Defaults to "f1" if present, otherwise
            the first available metric.
        ax: Optional matplotlib Axes to draw on. If None, a new figure is
            created.

    Returns:
        (fig, ax) — call plt.show() or fig.savefig() in the caller. If an ax
        was passed in, fig is the parent figure of that ax.
    """
    metric = _resolve_metric(cv_df, metric)
    score_col = f"mean_test_{metric}"

    def _full_col(name):
        if name.startswith("param_"):
            return name
        candidate = f"param_{name}"
        if candidate in cv_df.columns:
            return candidate
        raise ValueError(f"Column '{name}' or '{candidate}' not found in cv_df.")

    col_x = _full_col(param_x)
    col_y = _full_col(param_y)

    pivot = cv_df.groupby([col_y, col_x])[score_col].mean().unstack()
    pivot.index = pivot.index.astype(str)
    pivot.columns = pivot.columns.astype(str)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.5), max(3, len(pivot) * 1.2)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=_score_display_name(metric))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col_x.replace("param_", ""))
    ax.set_ylabel(col_y.replace("param_", ""))
    ax.set_title(f"Mean {_score_display_name(metric)}")

    for row_i in range(len(pivot.index)):
        for col_j in range(len(pivot.columns)):
            val = pivot.values[row_i, col_j]
            if not np.isnan(val):
                ax.text(col_j, row_i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    return fig, ax