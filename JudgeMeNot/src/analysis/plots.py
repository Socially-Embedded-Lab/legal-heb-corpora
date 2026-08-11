"""Plotting utilities for forest plots, CI plots, heatmaps, and ablation curves.

All plots use matplotlib and produce PDF/PNG outputs suitable for papers .
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def forest_plot(
    data: pd.DataFrame,
    delta_col: str = "delta",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    label_col: str = "model",
    title: str = "Forest Plot: Delta vs Baseline",
    output_path: Optional[str] = None,
):
    """Create a forest plot showing deltas with confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))

    y_pos = range(len(data))
    deltas = data[delta_col].values
    ci_low = data[ci_low_col].values
    ci_high = data[ci_high_col].values
    labels = data[label_col].values

    errors = np.array([deltas - ci_low, ci_high - deltas])
    colors = ["tab:blue" if d > 0 else "tab:red" for d in deltas]

    ax.errorbar(deltas, y_pos, xerr=errors, fmt="o", capsize=3,
                color="black", ecolor="gray", markersize=5)
    for i, (d, c) in enumerate(zip(deltas, colors)):
        ax.plot(d, i, "o", color=c, markersize=7, zorder=5)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta (own - others)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Forest plot saved: %s", output_path)
    return fig


def ablation_plot(
    data: pd.DataFrame,
    x_col: str = "percentage",
    y_cols: Optional[List[str]] = None,
    ci_suffix: str = "_ci_low",
    title: str = "Ablation Study",
    xlabel: str = "Data Size (%)",
    output_path: Optional[str] = None,
):
    """Plot ablation curves with optional CI bands."""
    if y_cols is None:
        y_cols = [c for c in data.columns if c.endswith("_mean") and c != x_col]

    n_metrics = len(y_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)

    x = data[x_col].values
    for i, col in enumerate(y_cols):
        ax = axes[0, i]
        y = data[col].values
        ax.plot(x, y, "o-", markersize=6)

        ci_low_col = col.replace("_mean", "_ci_low")
        ci_high_col = col.replace("_mean", "_ci_high")
        if ci_low_col in data.columns and ci_high_col in data.columns:
            ax.fill_between(x, data[ci_low_col], data[ci_high_col], alpha=0.2)

        metric_name = col.replace("_mean", "").replace("mean_", "")
        ax.set_title(metric_name)
        ax.set_xlabel(xlabel)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Ablation plot saved: %s", output_path)
    return fig


def heatmap(
    data: pd.DataFrame,
    title: str = "Results Heatmap",
    cmap: str = "RdYlGn",
    output_path: Optional[str] = None,
    annot: bool = True,
    fmt: str = ".3f",
):
    """Create a heatmap from a DataFrame (rows=models, cols=metrics)."""
    fig, ax = plt.subplots(figsize=(max(8, len(data.columns) * 1.2),
                                     max(4, len(data) * 0.5)))
    im = ax.imshow(data.values.astype(float), cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)

    if annot:
        for i in range(len(data)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Heatmap saved: %s", output_path)
    return fig


def ci_comparison_plot(
    data: pd.DataFrame,
    metric_col: str = "mean",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    label_col: str = "method",
    title: str = "Method Comparison",
    output_path: Optional[str] = None,
):
    """Bar chart with CI error bars comparing methods."""
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 5))

    x = range(len(data))
    means = data[metric_col].values
    ci_low = data[ci_low_col].values
    ci_high = data[ci_high_col].values
    errors = np.array([means - ci_low, ci_high - means])

    bars = ax.bar(x, means, yerr=errors, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(data[label_col], rotation=45, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("CI plot saved: %s", output_path)
    return fig


# ---------------------------------------------------------------------------
#  Cross-method forest plot (per-judge delta dots across metrics)
# ---------------------------------------------------------------------------

EXCLUDED_METRICS = {"perplexity", "dicta_bert_P", "dicta_bert_R", "kl_pos", "edit", "ppl", "rouge1", "rouge2"}

METRIC_DISPLAY = {
    "bleu": "BLEU",
    "jsd_pos": "JSD-POS",
    "bert_score_F": "BertScore-F1",
    "rougeL": "ROUGE-L",
    "dicta_bert_F": "BertScore-F1",
}

LOWER_IS_BETTER = {"jsd_pos", "JSD-POS"}


def cross_method_forest_plot(
    root_dir: str,
    filename: str = "results_own_data.csv",
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    bleu_scale: float = 100.0,
):
    """Forest plot showing per-judge delta dots across metrics.

    Reads metric subdirectories under root_dir, each containing a
    results_own_data.csv with columns: delta_boot_mean, delta_ci_lower,
    delta_ci_upper, is_significant, csv_model.

    Each metric becomes a row. Each judge is a dot positioned at its
    delta value, colored by significance.
    """
    root = Path(root_dir)
    all_data = []

    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name in EXCLUDED_METRICS:
            continue
        csv_path = item / filename
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            if "is_own_data" in df.columns:
                df = df[df["is_own_data"] == True]
            df["metric"] = item.name
            all_data.append(df)
        except Exception:
            continue

    if not all_data:
        logger.warning("No metric data found in %s", root_dir)
        return None

    combined = pd.concat(all_data, ignore_index=True)

    mask = combined["metric"] == "bleu"
    if mask.any() and bleu_scale > 1:
        for col in ["delta_boot_mean", "delta_ci_lower", "delta_ci_upper"]:
            combined.loc[mask, col] = combined.loc[mask, col] / bleu_scale

    combined["metric"] = combined["metric"].map(lambda m: METRIC_DISPLAY.get(m, m))
    combined["metric"] = combined["metric"].apply(
        lambda m: f"{m} ↓" if m in LOWER_IS_BETTER else f"{m} ↑"
    )

    metrics = sorted(combined["metric"].unique())
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (14, max(5, n_metrics * 0.6))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color_sig = "#1f77b4"
    color_ns = "#aec7e8"

    for idx, metric in enumerate(metrics):
        mdf = combined[combined["metric"] == metric].sort_values("delta_boot_mean")
        n = len(mdf)
        jitter = np.linspace(-0.1, 0.1, n) if n > 1 else np.array([0.0])

        for j, (_, row) in enumerate(mdf.iterrows()):
            is_sig = row.get("is_significant", False)
            ax.plot(
                row["delta_boot_mean"], idx + jitter[j], "o",
                color=color_sig if is_sig else color_ns,
                alpha=0.9 if is_sig else 0.5,
                markersize=8, markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            )

    ax.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8, zorder=1)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\Delta$", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.5, n_metrics - 0.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5, zorder=0)
    for i in range(n_metrics - 1):
        ax.axhline(y=i + 0.5, color="lightgray", linewidth=1.2, alpha=0.6, zorder=0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_sig,
               markersize=10, label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_ns,
               markersize=10, alpha=0.5, label="Not significant"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=2, label="Baseline (delta=0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        logger.info("Cross-method forest plot saved: %s", output_path)
    return fig


# ---------------------------------------------------------------------------
#  Cross-method summary table
# ---------------------------------------------------------------------------

def summarize_methods_metrics(
    base_dir: str,
    lower_is_better: Optional[set] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize (avg_delta, % significant) across methods and metrics.

    Expects directory structure: base_dir/<method>/<metric>/results_own_data.csv
    Returns a DataFrame with metrics as rows, methods as columns.
    """
    if lower_is_better is None:
        lower_is_better = {"perplexity", "jsd_pos", "kl_pos"}

    base = Path(base_dir)
    results = defaultdict(dict)

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        for metric_dir in sorted(method_dir.iterdir()):
            if not metric_dir.is_dir():
                continue
            csv_path = metric_dir / "results_own_data.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
                if "delta" not in df.columns:
                    continue
                avg_delta = df["delta"].mean()
                if metric_dir.name in lower_is_better:
                    avg_delta = -avg_delta
                n_total = len(df)
                n_sig = df["is_significant"].sum() if "is_significant" in df.columns else 0
                pct_sig = (n_sig / n_total * 100) if n_total > 0 else 0
                results[metric_dir.name][method_dir.name] = {
                    "avg_delta": avg_delta,
                    "pct_significant": pct_sig,
                }
            except Exception:
                continue

    all_methods = sorted({m for md in results.values() for m in md})
    rows = []
    for metric in sorted(results):
        row = {"metric": metric}
        for method in all_methods:
            if method in results[metric]:
                d = results[metric][method]
                row[method] = f"({d['avg_delta']:.4f}, {d['pct_significant']:.1f}%)"
            else:
                row[method] = "-"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("metric")
    if output_path:
        df.to_csv(output_path)
        logger.info("Method summary saved: %s", output_path)
    return df
