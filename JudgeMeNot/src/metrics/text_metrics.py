"""BLEU and ROUGE metrics with Hebrew UDPipe tokenization.

Adds per-row BLEU (sacrebleu sentence_bleu) and ROUGE1/2/L scores to
evaluation CSVs containing ground_truth and generated columns .
"""

from __future__ import annotations

import logging
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Hebrew tokenizer (lazy init per worker)
# ---------------------------------------------------------------------------

class HebrewTokenizer:
    """UDPipe-based Hebrew tokenizer for ROUGE computation."""

    def __init__(self, lang: str = "he"):
        import spacy_udpipe
        d = spacy_udpipe.utils.MODELS_DIR
        os.makedirs(d, exist_ok=True)
        if spacy_udpipe.utils.LANGUAGES[lang] not in os.listdir(d):
            spacy_udpipe.download(lang)
        self.nlp = spacy_udpipe.load(lang)

    def __call__(self, text: str) -> List[str]:
        return [tok.text for tok in self.nlp(text)]


# ---------------------------------------------------------------------------
#  Per-example metrics
# ---------------------------------------------------------------------------

def sentence_bleu(reference: str, hypothesis: str) -> float:
    import sacrebleu
    try:
        return sacrebleu.sentence_bleu(hypothesis, [reference]).score
    except Exception:
        return float("nan")


def rouge_fmeasures(
    refs: List[str],
    hyps: List[str],
    hebrew_tok: Optional[HebrewTokenizer] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute ROUGE-1/2/L f-measures per example."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rL = [], [], []

    for r, h in zip(refs, hyps):
        try:
            if hebrew_tok is not None:
                r = " ".join(hebrew_tok(r))
                h = " ".join(hebrew_tok(h))
            scores = scorer.score(r, h)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rL.append(scores["rougeL"].fmeasure)
        except Exception:
            r1.append(float("nan"))
            r2.append(float("nan"))
            rL.append(float("nan"))

    return r1, r2, rL


# ---------------------------------------------------------------------------
#  Batch CSV processing
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "ground_truth": ["answer", "reference", "target", "gt"],
    "generated": ["prediction", "pred", "predicted", "gen", "output"],
}


def _resolve_columns(df: pd.DataFrame, ref_col: str, pred_col: str):
    """Try aliases if expected columns are missing."""
    for desired, alts in _COLUMN_ALIASES.items():
        if desired not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: desired})
                    break
    return df


def add_metrics_to_csv(
    csv_path: str,
    ref_col: str = "ground_truth",
    pred_col: str = "generated",
    metrics: Optional[List[str]] = None,
    backup: bool = False,
) -> dict:
    """Add BLEU and ROUGE columns to a single CSV file in-place.

    Parameters
    ----------
    metrics : list of str, optional
        Which metrics to compute. Default: ["bleu", "rouge"].
    """
    if metrics is None:
        metrics = ["bleu", "rouge"]

    path = Path(csv_path)
    df = pd.read_csv(path)
    df = _resolve_columns(df, ref_col, pred_col)

    if not {ref_col, pred_col}.issubset(df.columns):
        return {"file": str(path), "status": "skipped", "reason": "missing columns"}

    refs = df[ref_col].fillna("").astype(str).tolist()
    hyps = df[pred_col].fillna("").astype(str).tolist()

    if "bleu" in metrics:
        df["bleu"] = [sentence_bleu(r, h) for r, h in zip(refs, hyps)]

    if "rouge" in metrics:
        try:
            hebrew_tok = HebrewTokenizer()
        except Exception:
            hebrew_tok = None
        r1, r2, rL = rouge_fmeasures(refs, hyps, hebrew_tok)
        df["rouge1"] = r1
        df["rouge2"] = r2
        df["rougeL"] = rL

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        path.rename(bak)

    df.to_csv(path, index=False)

    summary = {"file": str(path), "status": "ok", "rows": len(df)}
    for col in ["bleu", "rouge1", "rouge2", "rougeL"]:
        if col in df.columns:
            summary[f"{col}_mean"] = float(df[col].dropna().mean())
    return summary


def add_metrics_to_directory(
    input_dir: str,
    metrics: Optional[List[str]] = None,
    recursive: bool = False,
    workers: Optional[int] = None,
    ref_col: str = "ground_truth",
    pred_col: str = "generated",
) -> List[dict]:
    """Add metrics to all CSV files in a directory."""
    root = Path(input_dir)
    files = list(root.rglob("*.csv") if recursive else root.glob("*.csv"))
    if not files:
        logger.warning("No CSV files found in %s", root)
        return []

    n_workers = workers or min(cpu_count(), len(files))
    func = partial(add_metrics_to_csv, ref_col=ref_col, pred_col=pred_col, metrics=metrics)

    summaries = []
    if n_workers > 1 and len(files) > 1:
        with Pool(n_workers) as pool:
            for s in tqdm(pool.imap_unordered(func, [str(f) for f in files]),
                          total=len(files), desc="Adding metrics"):
                summaries.append(s)
    else:
        for f in tqdm(files, desc="Adding metrics"):
            summaries.append(func(str(f)))

    ok = sum(1 for s in summaries if s["status"] == "ok")
    logger.info("Processed %d / %d files", ok, len(files))
    return summaries
