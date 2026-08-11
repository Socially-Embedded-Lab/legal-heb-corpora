"""Fooling evaluation: test whether generated text can fool per-judge classifiers.

Loads trained DictaBERT classifiers and evaluates them on:
- Real test texts (positive = target judge's held-out text)
- Generated texts (negative = model-generated text for that judge)
- Optional random baseline and ground-truth (other judges') baseline

High classifier accuracy means the classifier easily separates real from generated
(generation did NOT fool it). Low accuracy near chance means the generated text
successfully mimics the judge's style.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Run classifier on a list of texts. Returns predicted labels (0/1)."""
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
    return np.array(all_preds)


# ---------------------------------------------------------------------------
#  CSV filename parsing (extract model and test judge)
# ---------------------------------------------------------------------------

_FILENAME_PATTERNS = [
    re.compile(r"(.+?)_model_vs_(.+?)_rag_data$"),
    re.compile(r"(.+?)_model_vs_(.+?)_test_data$"),
    re.compile(r"(.+?)_model_vs_(.+?)_data$"),
    re.compile(r"(.+?)_context_vs_(.+?)_test$"),
    re.compile(r"(.+?)_vs_(.+?)_test_data$"),
    re.compile(r"(.+?)_vs_(.+?)_test$"),
    re.compile(r"(.+?)_vs_(.+?)_data$"),
]


def parse_csv_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Extract (model_judge, test_judge) from evaluation CSV filename."""
    name = Path(filename).stem
    for pat in _FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            return m.group(1), m.group(2)
    if "_vs_" in name:
        parts = name.split("_vs_", 1)
        return parts[0], parts[1]
    return None


def _is_self_eval(model_name: str, test_name: str) -> bool:
    clean = lambda s: re.sub(r"[_\-\s]+", "", s.lower())
    return clean(model_name) == clean(test_name)


# ---------------------------------------------------------------------------
#  Load texts
# ---------------------------------------------------------------------------

def _load_json_texts(path: Path, label: Optional[int] = None) -> List[str]:
    """Load texts from a JSON file (list of dicts with 'text' and 'label')."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if label is not None:
        return [it["text"] for it in items if it.get("label") == label]
    return [it["text"] for it in items]


def _load_training_texts(dataset_dir: str, judge: str) -> set:
    """Load training texts for exclusion from evaluation."""
    path = Path(dataset_dir) / judge / f"{judge}_train.json"
    if not path.exists():
        return set()
    return {t.strip() for t in _load_json_texts(path)}


def collect_generated_texts(
    eval_dir: str,
    judge: str,
    self_only: bool = True,
    column: str = "generated",
    exclude_texts: Optional[set] = None,
) -> List[str]:
    """Collect generated texts for a judge from evaluation CSVs."""
    texts = []
    for csv_path in Path(eval_dir).glob("*.csv"):
        parsed = parse_csv_filename(csv_path.name)
        if parsed is None:
            continue
        model_j, test_j = parsed
        if self_only and not _is_self_eval(model_j, test_j):
            continue
        if not _is_self_eval(test_j, judge):
            continue
        try:
            df = pd.read_csv(csv_path)
            if column in df.columns:
                gen = df[column].dropna().astype(str).tolist()
                texts.extend(gen)
        except Exception:
            continue

    if exclude_texts:
        texts = [t for t in texts if t.strip() not in exclude_texts]
    return texts


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate_fooling(
    judge: str,
    classifier_dir: str,
    positive_texts: List[str],
    negative_texts: List[str],
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Evaluate how well a classifier distinguishes real vs generated text.

    Returns dict with accuracy, precision, recall, F1, and fool_rate.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(classifier_dir) / judge
    if not model_path.exists():
        return {"judge": judge, "status": "no_model"}

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device).eval()

    if max_samples:
        rng = np.random.default_rng(42)
        if len(positive_texts) > max_samples:
            idx = rng.choice(len(positive_texts), max_samples, replace=False)
            positive_texts = [positive_texts[i] for i in idx]
        if len(negative_texts) > max_samples:
            idx = rng.choice(len(negative_texts), max_samples, replace=False)
            negative_texts = [negative_texts[i] for i in idx]

    all_texts = positive_texts + negative_texts
    true_labels = np.array([1] * len(positive_texts) + [0] * len(negative_texts))

    preds = predict_batch(all_texts, model, tokenizer, batch_size, device=device)

    neg_preds = preds[len(positive_texts):]
    fool_rate = float(neg_preds.sum()) / max(len(neg_preds), 1)

    result = {
        "judge": judge,
        "n_positive": len(positive_texts),
        "n_negative": len(negative_texts),
        "accuracy": float(accuracy_score(true_labels, preds)),
        "precision": float(precision_score(true_labels, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(true_labels, preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(true_labels, preds, average="weighted", zero_division=0)),
        "fool_rate": fool_rate,
        "status": "ok",
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_fooling_evaluation(
    classifier_dir: str,
    eval_dir: str,
    dataset_dir: str,
    output_dir: str,
    method_name: str = "default",
    self_only: bool = True,
    max_samples: Optional[int] = None,
    include_random_baseline: bool = True,
    include_ground_truth: bool = True,
    batch_size: int = 32,
    device: str = "cuda",
    judges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run fooling evaluation for all judges against a generation method.

    For each judge:
    1. Real positive texts = held-out test texts (label 1) from dataset_dir
    2. Negative texts = generated texts from eval_dir CSVs
    3. Optional: random baseline (shuffled labels), ground-truth baseline (other judges' test texts)

    Returns results DataFrame and saves to output_dir/results.csv.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clf_root = Path(classifier_dir)
    if judges is None:
        judges = sorted(d.name for d in clf_root.iterdir() if d.is_dir() and (d / "config.json").exists())

    all_results = []

    for judge in tqdm(judges, desc=f"Fooling eval ({method_name})"):
        train_texts = _load_training_texts(dataset_dir, judge)

        test_json = Path(dataset_dir) / judge / f"{judge}_test.json"
        if test_json.exists():
            positives = _load_json_texts(test_json, label=1)
        else:
            positives = []

        if not positives:
            logger.warning("No positive test texts for %s, skipping", judge)
            continue

        generated = collect_generated_texts(
            eval_dir, judge, self_only=self_only, exclude_texts=train_texts,
        )
        if not generated:
            logger.warning("No generated texts for %s in %s, skipping", judge, eval_dir)
            continue

        res = evaluate_fooling(
            judge, classifier_dir, positives, generated,
            max_samples=max_samples, batch_size=batch_size, device=device,
        )
        res["method"] = method_name
        all_results.append(res)

        if include_random_baseline:
            rng = np.random.default_rng(42)
            n = min(len(positives), len(generated))
            pool = positives[:n] + generated[:n]
            rng.shuffle(pool)
            rand_res = evaluate_fooling(
                judge, classifier_dir, pool[:n], pool[n:],
                max_samples=max_samples, batch_size=batch_size, device=device,
            )
            rand_res["method"] = "random_baseline"
            all_results.append(rand_res)

        if include_ground_truth:
            other_test = []
            for other_dir in Path(dataset_dir).iterdir():
                if other_dir.name == judge or not other_dir.is_dir():
                    continue
                other_json = other_dir / f"{other_dir.name}_test.json"
                if other_json.exists():
                    other_test.extend(_load_json_texts(other_json, label=1))
            if other_test:
                gt_res = evaluate_fooling(
                    judge, classifier_dir, positives, other_test,
                    max_samples=max_samples, batch_size=batch_size, device=device,
                )
                gt_res["method"] = "ground_truth"
                all_results.append(gt_res)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out / "results.csv", index=False)

    summary = results_df.groupby("method").agg({
        "accuracy": "mean",
        "f1": "mean",
        "fool_rate": "mean",
    }).round(4)
    summary.to_csv(out / "summary.csv")
    logger.info("Fooling evaluation saved to %s", out)

    return results_df


def compute_fooling_significance(
    results_path: str,
    baseline_method: str = "random_baseline",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Wilcoxon + t-test on fooling results vs baseline."""
    from scipy.stats import ttest_rel, wilcoxon

    df = pd.read_csv(results_path)
    baseline = df[df["method"] == baseline_method].set_index("judge")
    methods = [m for m in df["method"].unique() if m != baseline_method]

    rows = []
    for method in methods:
        method_df = df[df["method"] == method].set_index("judge")
        common = sorted(set(baseline.index) & set(method_df.index))
        if len(common) < 3:
            continue
        b_vals = baseline.loc[common, "accuracy"].values
        m_vals = method_df.loc[common, "accuracy"].values

        try:
            w_stat, w_p = wilcoxon(m_vals, b_vals)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        try:
            t_stat, t_p = ttest_rel(m_vals, b_vals)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        rows.append({
            "method": method,
            "baseline": baseline_method,
            "n_judges": len(common),
            "method_acc_mean": float(m_vals.mean()),
            "baseline_acc_mean": float(b_vals.mean()),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "ttest_stat": float(t_stat),
            "ttest_p": float(t_p),
        })

    result = pd.DataFrame(rows)
    if output_path:
        result.to_csv(output_path, index=False)
    return result
