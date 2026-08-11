"""Classifier dataset preparation: per-judge binary datasets with length balancing.

Builds datasets where positive = texts from a target judge, negative = texts
pooled from all other judges. Supports local QA CSVs and local chunk CSVs
in the flat one-CSV-per-judge layout (each CSV has a ``split`` column).
Includes length-aware balancing and leakage-free splitting .
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

SEED = 42

DEFAULT_CHUNKS_ROOT = "data/chunks"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _get_length_bin(text: str, num_bins: int = 5) -> int:
    length = len(text)
    if length == 0:
        return 0
    return min(int(np.log1p(length) / np.log1p(10000) * num_bins), num_bins - 1)


def balance_by_length(
    positive: List[str],
    negative: List[str],
    target_size: Optional[int] = None,
    num_bins: int = 5,
) -> Tuple[List[str], List[str]]:
    """Length-balanced subsampling so positive and negative have similar length distributions."""
    if target_size is None:
        target_size = min(len(positive), len(negative))

    pos_bins: Dict[int, List[str]] = defaultdict(list)
    neg_bins: Dict[int, List[str]] = defaultdict(list)
    for t in positive:
        pos_bins[_get_length_bin(t, num_bins)].append(t)
    for t in negative:
        neg_bins[_get_length_bin(t, num_bins)].append(t)

    per_bin = max(1, target_size // num_bins)
    balanced_pos, balanced_neg = [], []
    rng = random.Random(SEED)

    for b in range(num_bins):
        p_pool, n_pool = pos_bins[b], neg_bins[b]
        n_take = min(per_bin, len(p_pool), len(n_pool))
        if n_take == 0:
            continue
        balanced_pos.extend(rng.sample(p_pool, n_take))
        balanced_neg.extend(rng.sample(n_pool, n_take))

    return balanced_pos, balanced_neg


def split_no_leakage(
    positive: List[str],
    negative: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[dict]]:
    """Split into train/val/test with no normalized-text overlap across splits."""
    rng = random.Random(SEED)

    def _dedup_and_shuffle(texts, label):
        seen = set()
        out = []
        for t in texts:
            norm = _normalize_text(t)
            if norm not in seen:
                seen.add(norm)
                out.append({"text": t, "label": label})
        rng.shuffle(out)
        return out

    pos = _dedup_and_shuffle(positive, 1)
    neg = _dedup_and_shuffle(negative, 0)

    def _split_list(items):
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    p_tr, p_va, p_te = _split_list(pos)
    n_tr, n_va, n_te = _split_list(neg)

    def _merge_shuffle(a, b):
        combined = a + b
        rng.shuffle(combined)
        return combined

    return {
        "train": _merge_shuffle(p_tr, n_tr),
        "validation": _merge_shuffle(p_va, n_va),
        "test": _merge_shuffle(p_te, n_te),
    }


# ---------------------------------------------------------------------------
#  Data sources (flat CSV layout)
# ---------------------------------------------------------------------------

def load_local_judge_texts(data_root: str) -> Dict[str, List[str]]:
    """Load per-judge texts from flat QA CSVs (answer column).

    Expects ``data_root/*.csv`` where each CSV has ``answer`` and ``split``
    columns.
    """
    root = Path(data_root)
    judges: Dict[str, List[str]] = {}
    for csv_path in sorted(root.glob("*.csv")):
        if csv_path.stem == "leakage_report":
            continue
        try:
            df = pd.read_csv(csv_path)
            if "answer" in df.columns:
                texts = df["answer"].dropna().astype(str).tolist()
                if texts:
                    judges[csv_path.stem] = texts
        except Exception:
            continue
    logger.info("Loaded %d judges from local data", len(judges))
    return judges


def load_chunks_judge_texts(
    chunks_root: Optional[str] = None,
    splits: Tuple[str, ...] = ("train", "eval"),
) -> Dict[str, List[str]]:
    """Load per-judge texts from flat chunk CSVs.

    Expects ``chunks_root/*.csv`` where each CSV has ``text`` and ``split``
    columns.
    """
    root = Path(chunks_root or DEFAULT_CHUNKS_ROOT)
    if not root.exists():
        logger.warning("Chunks root %s not found", root)
        return {}

    judges: Dict[str, List[str]] = {}
    for csv_path in tqdm(sorted(root.glob("*.csv")), desc="Loading chunk CSVs"):
        if csv_path.stem == "leakage_report":
            continue
        try:
            df = pd.read_csv(csv_path)
            col = "text" if "text" in df.columns else df.columns[0]
            if "split" in df.columns:
                df = df[df["split"].isin(splits)]
            texts = df[col].dropna().astype(str).tolist()
            if texts:
                judges[csv_path.stem] = texts
        except Exception:
            continue
    logger.info("Loaded %d judges from local chunks", len(judges))
    return judges


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def prepare_judge_dataset(
    judge: str,
    all_judges_data: Dict[str, List[str]],
    max_samples: int = 5000,
    num_bins: int = 5,
) -> Tuple[Dict[str, List[dict]], dict]:
    """Prepare a balanced, leakage-free binary dataset for one judge.

    Returns (splits_dict, metadata_dict).
    """
    positive = all_judges_data.get(judge, [])
    negative = []
    for other, texts in all_judges_data.items():
        if other != judge:
            negative.extend(texts)

    target = min(max_samples, len(positive), len(negative))
    bal_pos, bal_neg = balance_by_length(positive, negative, target, num_bins)
    splits = split_no_leakage(bal_pos, bal_neg)

    metadata = {
        "judge": judge,
        "total_positive": len(positive),
        "total_negative": len(negative),
        "balanced_size": len(bal_pos),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
    }
    return splits, metadata


def prepare_all_judges(
    data_root: Optional[str] = None,
    chunks_root: Optional[str] = None,
    output_dir: str = "outputs/classifier_datasets",
    max_samples: int = 5000,
) -> pd.DataFrame:
    """Prepare classifier datasets for all judges.

    Uses local QA CSVs if data_root is given, otherwise local chunk CSVs.
    Saves per-judge JSON splits + summary CSV.
    """
    if data_root:
        all_data = load_local_judge_texts(data_root)
    else:
        all_data = load_chunks_judge_texts(chunks_root)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []

    for judge in tqdm(sorted(all_data.keys()), desc="Preparing datasets"):
        judge_dir = out / judge
        judge_dir.mkdir(exist_ok=True)

        splits, meta = prepare_judge_dataset(judge, all_data, max_samples)

        for split_name, items in splits.items():
            with open(judge_dir / f"{judge}_{split_name}.json", "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

        with open(judge_dir / f"{judge}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summaries.append(meta)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "datasets_summary.csv", index=False)
    logger.info("Saved datasets for %d judges to %s", len(summaries), out)
    return summary_df
