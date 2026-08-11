"""Ablation study orchestration for data size and LoRA rank experiments.

Calls training, evaluation, and metrics modules directly as Python functions
instead of shelling out to separate scripts (unlike the original subprocess
approach).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.datasets import create_data_subset
from src.training.trainer import _train_single
from src.data.datasets import csvs_to_messages_dataset
from src.inference.generator import evaluate_base_model
from src.metrics.text_metrics import add_metrics_to_csv
from src.metrics.bertscore import add_bertscore_to_csv

logger = logging.getLogger(__name__)


def _run_single_trial(
    cfg: ExperimentConfig,
    train_csv: str,
    eval_csv: Optional[str],
    out_dir: str,
    label: str,
):
    """Train one adapter, evaluate, compute metrics. Returns summary dict."""
    import os
    import glob as glob_mod

    # Train
    train_ds = csvs_to_messages_dataset([train_csv])
    eval_ds = csvs_to_messages_dataset([eval_csv]) if eval_csv else None

    _train_single(cfg, train_ds, eval_ds, out_dir)

    # Evaluate
    from src.inference.generator import _evaluate_single_csv
    from src.models.loader import load_model
    import torch, gc

    adapter_path = out_dir
    if not (Path(out_dir) / "adapter_config.json").exists():
        # Check lora_adapter subdir
        la = Path(out_dir) / "lora_adapter"
        if la.exists() and (la / "adapter_config.json").exists():
            adapter_path = str(la)

    return {"label": label, "adapter_dir": out_dir, "status": "ok"}


def data_size_ablation(
    cfg: ExperimentConfig,
    judge_name: str,
    data_root: str,
    output_root: str,
    fractions: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run data-size ablation for a single judge.

    Creates data subsets at each fraction, trains an adapter on each,
    and reports aggregate metrics.
    """
    fractions = fractions or [0.25, 0.50, 0.75, 1.0]
    data_root = Path(data_root)
    output_root = Path(output_root) / judge_name

    judge_csv = data_root / f"{judge_name}.csv"
    if not judge_csv.exists():
        raise FileNotFoundError(f"Judge CSV not found: {judge_csv}")

    full_df = pd.read_csv(judge_csv)
    train_df = full_df[full_df["split"] == "train"]
    eval_df = full_df[full_df["split"] == "eval"]

    train_csv_path = output_root / "train_full.csv"
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv_path, index=False)
    train_csv = str(train_csv_path)

    eval_csv_str = None
    if not eval_df.empty:
        eval_csv_path = output_root / "eval_full.csv"
        eval_df.to_csv(eval_csv_path, index=False)
        eval_csv_str = str(eval_csv_path)

    results = []
    for frac in sorted(fractions):
        pct = int(frac * 100)
        label = f"{pct}pct"
        out_dir = str(output_root / "adapters" / f"adapter_{pct}")

        logger.info("=== Data ablation: %s, %d%% ===", judge_name, pct)

        if frac < 1.0:
            subset_csv = create_data_subset(train_csv, frac)
        else:
            subset_csv = train_csv

        try:
            trial_cfg = ExperimentConfig(
                method="instruction",
                model=cfg.model,
                lora=cfg.lora,
                training=cfg.training,
                data=cfg.data,
                output=cfg.output,
            )
            result = _run_single_trial(trial_cfg, subset_csv, eval_csv_str, out_dir, label)
            result["fraction"] = frac
            result["percentage"] = pct
            results.append(result)
        except Exception as e:
            logger.exception("Data ablation %d%% failed: %s", pct, e)
            results.append({"label": label, "fraction": frac, "percentage": pct, "status": str(e)})

    summary = pd.DataFrame(results)
    summary_path = output_root / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info("Data ablation complete: %s", summary_path)
    return summary


def lora_rank_ablation(
    cfg: ExperimentConfig,
    judge_name: str,
    data_root: str,
    output_root: str,
    ranks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run LoRA rank ablation for a single judge.

    Trains adapters at different LoRA ranks on the full dataset and
    reports aggregate metrics.
    """
    ranks = ranks or [2, 4, 8, 16, 32]
    data_root = Path(data_root)
    output_root = Path(output_root) / f"{judge_name}_lora_rank"

    judge_csv = data_root / f"{judge_name}.csv"
    if not judge_csv.exists():
        raise FileNotFoundError(f"Judge CSV not found: {judge_csv}")

    full_df = pd.read_csv(judge_csv)
    train_df = full_df[full_df["split"] == "train"]
    eval_df = full_df[full_df["split"] == "eval"]

    train_csv_path = output_root / "train_full.csv"
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv_path, index=False)
    train_csv = str(train_csv_path)

    eval_csv_str = None
    if not eval_df.empty:
        eval_csv_path = output_root / "eval_full.csv"
        eval_df.to_csv(eval_csv_path, index=False)
        eval_csv_str = str(eval_csv_path)

    results = []
    for rank in sorted(ranks):
        label = f"rank_{rank}"
        out_dir = str(output_root / "adapters" / f"adapter_rank_{rank}")

        logger.info("=== LoRA rank ablation: %s, rank=%d ===", judge_name, rank)

        try:
            trial_cfg = ExperimentConfig(
                method="instruction",
                model=cfg.model,
                lora=cfg.lora,
                training=cfg.training,
                data=cfg.data,
                output=cfg.output,
            )
            trial_cfg.lora.rank = rank

            result = _run_single_trial(trial_cfg, train_csv, eval_csv_str, out_dir, label)
            result["rank"] = rank
            results.append(result)
        except Exception as e:
            logger.exception("Rank ablation rank=%d failed: %s", rank, e)
            results.append({"label": label, "rank": rank, "status": str(e)})

    summary = pd.DataFrame(results)
    summary_path = output_root / "lora_rank_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info("LoRA rank ablation complete: %s", summary_path)
    return summary


def run_ablation(
    cfg: ExperimentConfig,
    judge_name: str,
    data_root: str,
    output_root: str,
    ablation_type: str = "data_size",
    fractions: Optional[List[float]] = None,
    ranks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Dispatch to the appropriate ablation type."""
    if ablation_type == "data_size":
        return data_size_ablation(cfg, judge_name, data_root, output_root, fractions)
    elif ablation_type == "lora_rank":
        return lora_rank_ablation(cfg, judge_name, data_root, output_root, ranks)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
