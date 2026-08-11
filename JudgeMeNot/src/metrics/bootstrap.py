"""Bootstrap confidence intervals and delta analysis.

Computes bootstrap mean/CI for evaluation metrics and significance of
deltas between personalized (own-judge) and cross-judge (others) results .
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "bleu", "rouge1", "rouge2", "rougeL",
    "dicta_bert_F", "dicta_bert_P", "dicta_bert_R",
    "ft_bert_F",
    "perplexity",
    "jsd_pos",
]


def bootstrap_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap mean with confidence interval.

    Returns dict with keys: mean, ci_low, ci_high, std.
    """
    rng = np.random.RandomState(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}

    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(means, alpha * 100)),
        "ci_high": float(np.percentile(means, (1 - alpha) * 100)),
        "std": float(np.std(values)),
    }


def bootstrap_delta(
    own_values: np.ndarray,
    others_values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap delta (own - others) with CI and significance."""
    rng = np.random.RandomState(seed)

    own_clean = own_values[~np.isnan(own_values)]
    others_clean = others_values[~np.isnan(others_values)]
    if len(own_clean) == 0 or len(others_clean) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "significant": False}

    deltas = []
    for _ in range(n_boot):
        own_sample = rng.choice(own_clean, size=len(own_clean), replace=True)
        others_sample = rng.choice(others_clean, size=len(others_clean), replace=True)
        deltas.append(np.mean(own_sample) - np.mean(others_sample))

    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(deltas, alpha * 100))
    ci_high = float(np.percentile(deltas, (1 - alpha) * 100))

    return {
        "delta": float(np.mean(own_clean) - np.mean(others_clean)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


def compute_bootstrap_summary(
    input_dir: str,
    output_path: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute bootstrap mean/CI for each metric across all CSVs in a directory.

    Each CSV file contributes one row to the summary (aggregated across all
    examples in that file).
    """
    root = Path(input_dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        logger.warning("No CSVs in %s", root)
        return pd.DataFrame()

    if metrics is None:
        metrics = METRIC_COLUMNS

    rows = []
    for f in tqdm(files, desc="Bootstrap summary"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        row = {"file": f.name}
        for col in metrics:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values.astype(float)
            stats = bootstrap_mean(vals, n_boot=n_boot, ci=ci)
            for k, v in stats.items():
                row[f"{col}_{k}"] = v
        rows.append(row)

    summary = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info("Bootstrap summary: %s (%d files)", output_path, len(rows))
    return summary


def compute_delta_analysis(
    eval_dir: str,
    output_path: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute own-vs-others delta with bootstrap CIs.

    Expects eval CSVs named: MODEL_model_vs_DATASET_data.csv
    "Own" = model name matches dataset judge; "Others" = doesn't match.
    """
    root = Path(eval_dir)
    files = sorted(root.glob("*_model_vs_*_data.csv"))

    if metrics is None:
        metrics = METRIC_COLUMNS

    # Group by model
    model_data: Dict[str, Dict[str, list]] = {}
    for f in files:
        parts = f.stem.split("_model_vs_")
        if len(parts) != 2:
            continue
        model_name = parts[0]
        dataset_name = parts[1].replace("_data", "")

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        is_own = model_name.lower().replace("_", "") in dataset_name.lower().replace("_", "")
        key = "own" if is_own else "others"

        if model_name not in model_data:
            model_data[model_name] = {"own": [], "others": []}

        for col in metrics:
            if col in df.columns:
                model_data[model_name][key].extend(df[col].dropna().tolist())

    rows = []
    for model_name, data in model_data.items():
        own = np.array(data["own"], dtype=float)
        others = np.array(data["others"], dtype=float)
        if len(own) == 0 or len(others) == 0:
            continue

        row = {"model": model_name}
        own_stats = bootstrap_mean(own, n_boot, ci)
        others_stats = bootstrap_mean(others, n_boot, ci)
        delta_stats = bootstrap_delta(own, others, n_boot, ci)

        for k, v in own_stats.items():
            row[f"own_{k}"] = v
        for k, v in others_stats.items():
            row[f"others_{k}"] = v
        for k, v in delta_stats.items():
            row[f"delta_{k}"] = v
        rows.append(row)

    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Delta analysis: %s (%d models)", output_path, len(rows))
    return result
