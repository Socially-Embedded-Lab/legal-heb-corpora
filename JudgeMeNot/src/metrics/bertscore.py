"""BERTScore computation using DictaBERT (Hebrew) and optional fine-tuned checkpoints.

Processes CSV files by adding BERTScore precision/recall/F1 columns in-place .
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from bert_score import score as bert_score_fn

logger = logging.getLogger(__name__)


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    model_type: str = "dicta-il/dictabert",
    num_layers: int = 12,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> tuple:
    """Compute BERTScore and return (P, R, F1) tensors."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score_fn(
        hypotheses, references,
        model_type=model_type,
        num_layers=num_layers,
        lang="he",
        idf=False,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return P, R, F1


def add_bertscore_to_csv(
    csv_path: str,
    ref_col: str = "ground_truth",
    pred_col: str = "generated",
    model_type: str = "dicta-il/dictabert",
    ft_model_path: Optional[str] = None,
    num_layers: int = 12,
    device: Optional[str] = None,
    prefix: str = "dicta_bert",
) -> dict:
    """Add BERTScore columns to a CSV file in-place.

    Columns added: {prefix}_P, {prefix}_R, {prefix}_F
    If ft_model_path is provided, also adds ft_bert_P/R/F.
    """
    path = Path(csv_path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"file": str(path), "status": "error", "reason": str(e)}

    if ref_col not in df.columns or pred_col not in df.columns:
        return {"file": str(path), "status": "skipped", "reason": "missing columns"}

    refs = df[ref_col].astype(str).tolist()
    hyps = df[pred_col].astype(str).tolist()

    try:
        P, R, F1 = compute_bertscore(hyps, refs, model_type, num_layers, device)
        df[f"{prefix}_P"] = P.tolist()
        df[f"{prefix}_R"] = R.tolist()
        df[f"{prefix}_F"] = F1.tolist()
    except Exception as e:
        logger.error("BERTScore failed for %s: %s", path.name, e)
        return {"file": str(path), "status": "error", "reason": str(e)}

    if ft_model_path:
        try:
            P2, R2, F2 = compute_bertscore(hyps, refs, ft_model_path, num_layers, device)
            df["ft_bert_P"] = P2.tolist()
            df["ft_bert_R"] = R2.tolist()
            df["ft_bert_F"] = F2.tolist()
        except Exception as e:
            logger.warning("Fine-tuned BERTScore failed: %s", e)

    df.to_csv(path, index=False)
    logger.info("Updated %s with BERTScore", path.name)
    return {"file": str(path), "status": "ok", "rows": len(df)}


def add_bertscore_to_directory(
    input_dir: str,
    ref_col: str = "ground_truth",
    pred_col: str = "generated",
    model_type: str = "dicta-il/dictabert",
    ft_model_path: Optional[str] = None,
    num_layers: int = 12,
    device: Optional[str] = None,
) -> List[dict]:
    """Add BERTScore to all CSV files in a directory."""
    from tqdm.auto import tqdm

    root = Path(input_dir)
    files = [f for f in root.glob("*.csv")]
    results = []

    for f in tqdm(files, desc="BERTScore"):
        # Check columns before loading GPU model
        try:
            header = pd.read_csv(f, nrows=0)
            if ref_col not in header.columns or pred_col not in header.columns:
                results.append({"file": str(f), "status": "skipped"})
                continue
        except Exception:
            continue

        r = add_bertscore_to_csv(
            str(f), ref_col, pred_col, model_type, ft_model_path, num_layers, device,
        )
        results.append(r)

    return results
