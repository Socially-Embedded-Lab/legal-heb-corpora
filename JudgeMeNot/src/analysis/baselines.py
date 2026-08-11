"""Wilcoxon signed-rank tests for baseline model comparisons.

Compares a reference model (e.g. clm-4b) against all other models using
paired Wilcoxon tests on per-judge aggregate metrics .
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


def wilcoxon_comparison(
    results_csv: str,
    reference_model: str = "clm-4b",
    score_col: str = "regular_mean",
    test_col: str = "test",
    model_col: str = "model",
    lower_is_better: bool = False,
) -> pd.DataFrame:
    """Compare reference model against all others using paired Wilcoxon tests.

    Parameters
    ----------
    results_csv : str
        Path to results_long.csv with columns: model, test, regular_mean.
    reference_model : str
        Model name to use as reference (must exist in the model column).
    lower_is_better : bool
        If True, negative delta means reference is better.

    Returns
    -------
    DataFrame with columns: model, mean_delta, median_delta, pct_ref_better,
    wilcoxon_pvalue, n_judges.
    """
    df = pd.read_csv(results_csv)

    ref_data = (
        df[df[model_col] == reference_model][[test_col, score_col]]
        .set_index(test_col)[score_col]
    )
    other_models = [m for m in df[model_col].unique() if m != reference_model]

    results = []
    for other in other_models:
        other_data = (
            df[df[model_col] == other][[test_col, score_col]]
            .set_index(test_col)[score_col]
        )
        common = ref_data.index.intersection(other_data.index)
        if len(common) == 0:
            continue

        ref_scores = ref_data[common].values
        other_scores = other_data[common].values
        deltas = ref_scores - other_scores

        if lower_is_better:
            pct_better = 100 * np.sum(deltas < 0) / len(deltas)
        else:
            pct_better = 100 * np.sum(deltas > 0) / len(deltas)

        try:
            stat, p_value = wilcoxon(ref_scores, other_scores, alternative="two-sided")
        except Exception:
            p_value = float("nan")

        results.append({
            "model": other,
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "pct_ref_better": float(pct_better),
            "wilcoxon_pvalue": float(p_value),
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
            "n_judges": len(common),
        })

    return pd.DataFrame(results).sort_values("mean_delta", ascending=False)


def cross_method_comparison(
    results_csv: str,
    model_pairs: Optional[List[tuple]] = None,
    score_col: str = "regular_mean",
    test_col: str = "test",
    model_col: str = "model",
) -> pd.DataFrame:
    """Cross-method Wilcoxon grid (e.g., 1B vs 4B for each method).

    Parameters
    ----------
    model_pairs : list of (model_a, model_b) tuples
        If None, compares all unique pairs.
    """
    df = pd.read_csv(results_csv)
    models = sorted(df[model_col].unique())

    if model_pairs is None:
        model_pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]

    results = []
    for m_a, m_b in model_pairs:
        data_a = df[df[model_col] == m_a][[test_col, score_col]].set_index(test_col)[score_col]
        data_b = df[df[model_col] == m_b][[test_col, score_col]].set_index(test_col)[score_col]
        common = data_a.index.intersection(data_b.index)
        if len(common) < 3:
            continue

        a_vals = data_a[common].values
        b_vals = data_b[common].values

        try:
            stat, p_value = wilcoxon(a_vals, b_vals, alternative="two-sided")
        except Exception:
            p_value = float("nan")

        results.append({
            "model_a": m_a,
            "model_b": m_b,
            "mean_a": float(np.mean(a_vals)),
            "mean_b": float(np.mean(b_vals)),
            "mean_delta": float(np.mean(a_vals - b_vals)),
            "wilcoxon_pvalue": float(p_value),
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
            "n_judges": len(common),
        })

    return pd.DataFrame(results)


def run_baseline_analysis(
    input_dir: str,
    output_dir: Optional[str] = None,
    reference_model: str = "clm-4b",
    lower_is_better: bool = False,
):
    """Run Wilcoxon analysis on a directory containing results_long.csv."""
    input_dir = Path(input_dir)
    results_csv = input_dir / "results_long.csv"

    if not results_csv.exists():
        logger.error("results_long.csv not found in %s", input_dir)
        return

    out_dir = Path(output_dir) if output_dir else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = wilcoxon_comparison(
        str(results_csv),
        reference_model=reference_model,
        lower_is_better=lower_is_better,
    )

    out_path = out_dir / "wilcoxon_comparison_results.csv"
    results.to_csv(out_path, index=False)
    logger.info("Wilcoxon results: %s", out_path)

    print(results.to_string(index=False))
    print(f"\nSignificant (p<0.05): {results['significant'].sum()}")
    return results
