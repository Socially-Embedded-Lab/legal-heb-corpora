"""POS-tag distribution metrics (Jensen-Shannon divergence).

Uses DictaBERT parser for Hebrew POS tagging and compares distributions
between reference and generated texts .
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_parser():
    """Lazy-load the DictaBERT parser pipeline."""
    from transformers import pipeline
    return pipeline("token-classification", model="dicta-il/dictabert-parse", device=-1)


def get_pos_distribution(text: str, parser) -> Dict[str, float]:
    """Get normalized POS tag distribution for a text."""
    try:
        tokens = parser(text)
        pos_counts = Counter(t["entity"] for t in tokens if t.get("entity"))
        total = sum(pos_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in pos_counts.items()}
    except Exception:
        return {}


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """KL(P || Q) with smoothing."""
    all_tags = set(p.keys()) | set(q.keys())
    if not all_tags:
        return 0.0
    kl = 0.0
    for tag in all_tags:
        p_val = p.get(tag, epsilon)
        q_val = q.get(tag, epsilon)
        kl += p_val * np.log(p_val / q_val)
    return float(kl)


def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    all_tags = set(p.keys()) | set(q.keys())
    if not all_tags:
        return 0.0
    m = {tag: 0.5 * (p.get(tag, 0.0) + q.get(tag, 0.0)) for tag in all_tags}
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def add_style_metrics_to_csv(
    csv_path: str,
    ref_col: str = "ground_truth",
    pred_col: str = "generated",
) -> dict:
    """Add POS-distribution JSD column to a CSV."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    if ref_col not in df.columns or pred_col not in df.columns:
        return {"file": str(path), "status": "skipped"}

    parser = _load_parser()
    jsd_scores = []

    for _, row in df.iterrows():
        ref_dist = get_pos_distribution(str(row[ref_col]), parser)
        gen_dist = get_pos_distribution(str(row[pred_col]), parser)
        jsd_scores.append(jsd(ref_dist, gen_dist))

    df["jsd_pos"] = jsd_scores
    df.to_csv(path, index=False)

    return {
        "file": str(path),
        "status": "ok",
        "jsd_mean": float(np.mean(jsd_scores)),
    }
