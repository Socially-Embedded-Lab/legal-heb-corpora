"""Unified LoRA SFT training for CLM, Instruction Tuning, and CLORA.

The three methods differ only in data formatting:
  - CLM:         text + EOS continuation (local CSV or HF datasets)
  - Instruction: question/answer CSV -> chat messages
  - CLORA:       Same as Instruction but base loaded from pre-merged directory

The SFTTrainer, LoraConfig, early stopping, checkpointing, and cleanup
logic is identical across all three.
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

from src.config import ExperimentConfig
from src.data.datasets import (
    csvs_to_messages_dataset,
    discover_judges,
    load_clm_dataset,
    load_judge_csvs,
)
from src.models.loader import load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  LoRA + SFTConfig builders
# ---------------------------------------------------------------------------

def _build_lora_config(cfg: ExperimentConfig) -> PeftLoraConfig:
    target = cfg.lora.target_modules
    if isinstance(target, str) and target != "all-linear":
        target = [t.strip() for t in target.split(",")]

    return PeftLoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=target,
        bias="none",
    )


def _build_sft_config(
    cfg: ExperimentConfig,
    out_dir: str,
    has_eval: bool,
) -> SFTConfig:
    tc = cfg.training
    is_clm = cfg.method == "clm"

    common = dict(
        max_length=cfg.model.seq_length,
        packing=tc.packing,
        num_train_epochs=tc.epochs if tc.max_steps <= 0 else 1,
        max_steps=tc.max_steps if tc.max_steps > 0 else -1,
        per_device_train_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation,
        gradient_checkpointing=True,
        optim=tc.optim,
        max_grad_norm=tc.max_grad_norm,
        warmup_ratio=tc.warmup_ratio,
        warmup_steps=tc.warmup_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        lr_scheduler_type=tc.scheduler,
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=out_dir,
        report_to="tensorboard",
        logging_dir=os.path.join(out_dir, "runs"),
        seed=tc.seed,
        save_only_model=True,
    )

    if not is_clm:
        common["dataset_kwargs"] = {
            "add_special_tokens": False,
            "append_concat_token": True,
        }
    else:
        common["dataset_text_field"] = "text"

    if has_eval:
        return SFTConfig(
            eval_strategy="steps",
            eval_steps=tc.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            **common,
        )
    return SFTConfig(
        eval_strategy="no",
        load_best_model_at_end=False,
        save_total_limit=1,
        **common,
    )


# ---------------------------------------------------------------------------
#  Core training loop
# ---------------------------------------------------------------------------

def _train_single(
    cfg: ExperimentConfig,
    train_ds,
    eval_ds,
    out_dir: str,
    base_model_path: Optional[str] = None,
):
    """Train one LoRA adapter and save it.

    Parameters
    ----------
    base_model_path : str, optional
        Override for the base model (used by CLORA for merged bases).
    """
    model_path = base_model_path or cfg.model.name

    tokenizer, model = load_model(
        model_path,
        quantize=cfg.model.quantize,
        is_trainable=True,
        attn_implementation=cfg.model.attn_implementation,
    )

    if cfg.method == "clm":
        model.resize_token_embeddings(len(tokenizer))
        prepare_model_for_kbit_training(model)

    lora_cfg = _build_lora_config(cfg)
    sft_cfg = _build_sft_config(cfg, out_dir, eval_ds is not None)

    callbacks = None
    if eval_ds is not None:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.patience,
                early_stopping_threshold=0.0,
            )
        ]

    # Gemma memory tweak: force micro-batch 1
    if "gemma" in model_path.lower() and sft_cfg.per_device_train_batch_size > 1:
        logger.warning("Gemma detected - forcing micro-batch=1, adjusting grad_accum")
        sft_cfg.gradient_accumulation_steps *= sft_cfg.per_device_train_batch_size
        sft_cfg.per_device_train_batch_size = 1

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(out_dir)
    logger.info("Saved adapter to %s", out_dir)

    # Save resolved config for reproducibility
    with open(os.path.join(out_dir, "resolved_config.json"), "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    # Cleanup checkpoints
    for ckpt in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

    del trainer, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
#  Public entry points
# ---------------------------------------------------------------------------

def train_clm(cfg: ExperimentConfig):
    """Train per-dataset LoRA adapters using causal LM (text continuation)."""
    if not cfg.data.datasets:
        raise ValueError("CLM training requires data.datasets (list of dataset names)")

    output_root = Path(cfg.output.root)

    for ds_name in cfg.data.datasets:
        slug = ds_name.split("/")[-1].replace("-", "_")
        out_dir = str(output_root / slug)

        if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
            logger.info("Skipping %s (already trained)", ds_name)
            continue

        os.makedirs(out_dir, exist_ok=True)
        logger.info("Training CLM on %s", ds_name)

        tokenizer, _ = load_model(cfg.model.name, quantize="none", device_map="cpu")
        ds = load_clm_dataset(
            ds_name, tokenizer.eos_token, chunks_root=cfg.data.chunks_root,
        )
        del tokenizer
        torch.cuda.empty_cache()

        _train_single(cfg, ds["train"], ds.get("eval"), out_dir)


def train_instruction(cfg: ExperimentConfig):
    """Train per-judge LoRA adapters using instruction tuning."""
    data_root = Path(cfg.data.root)
    output_root = Path(cfg.output.root)

    judge_csvs = discover_judges(data_root)
    if not judge_csvs:
        raise ValueError(f"No judge CSVs found under {data_root}")

    for judge_csv in judge_csvs:
        judge_name = judge_csv.stem
        out_dir = str(output_root / judge_name)

        if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
            logger.info("Skipping %s (already trained)", judge_name)
            continue

        os.makedirs(out_dir, exist_ok=True)
        logger.info("Training IT on judge: %s", judge_name)

        train_ds, eval_ds = load_judge_csvs(judge_csv)
        if train_ds is None:
            logger.warning("No train data for %s, skipping", judge_name)
            continue

        _train_single(cfg, train_ds, eval_ds, out_dir)


def train_clora(cfg: ExperimentConfig):
    """Train LoRA adapters on pre-merged base models (CLORA)."""
    if not cfg.data.merged_root:
        raise ValueError("CLORA training requires data.merged_root")

    merged_root = Path(cfg.data.merged_root)
    data_root = Path(cfg.data.root)
    output_root = Path(cfg.output.root)

    model_dirs = [d for d in sorted(merged_root.iterdir()) if d.is_dir()]
    if not model_dirs:
        raise ValueError(f"No model directories under {merged_root}")

    for mdir in model_dirs:
        judge_csv = data_root / f"{mdir.name}.csv"
        if not judge_csv.exists():
            logger.warning("No dataset for %s, skipping", mdir.name)
            continue

        out_dir = str(output_root / mdir.name / "lora_adapter")

        if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
            logger.info("Skipping %s (already trained)", mdir.name)
            continue

        os.makedirs(out_dir, exist_ok=True)
        logger.info("Training CLORA on merged model: %s", mdir.name)

        train_ds, eval_ds = load_judge_csvs(judge_csv)
        if train_ds is None:
            logger.warning("No train data for %s, skipping", mdir.name)
            continue

        _train_single(cfg, train_ds, eval_ds, out_dir, base_model_path=str(mdir))


def train(cfg: ExperimentConfig):
    """Dispatch to the appropriate training method based on config."""
    method = cfg.method
    logger.info("Starting training: method=%s, model=%s", method, cfg.model.name)

    if method == "clm":
        train_clm(cfg)
    elif method == "instruction":
        train_instruction(cfg)
    elif method == "clora":
        train_clora(cfg)
    else:
        raise ValueError(f"Unknown training method: {method}")


# ---------------------------------------------------------------------------
#  Hyperparameter search
# ---------------------------------------------------------------------------

def hparam_search(
    cfg: ExperimentConfig,
    train_csv: str,
    eval_csv: Optional[str] = None,
    lora_ranks: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None,
    output_root: str = "outputs/hparam_search",
):
    """Grid search over LoRA ranks and learning rates.

    Trains one adapter per combination and logs results to a summary CSV.
    """
    import pandas as pd

    ranks = lora_ranks or [8, 16, 32]
    lrs = learning_rates or [1e-4, 2e-4, 5e-4]
    results = []

    train_csvs = [train_csv]
    train_ds = csvs_to_messages_dataset(train_csvs)
    eval_ds = csvs_to_messages_dataset([eval_csv]) if eval_csv else None

    for rank in ranks:
        for lr in lrs:
            trial_name = f"r{rank}_lr{lr}"
            out_dir = os.path.join(output_root, trial_name)

            if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
                logger.info("Skipping %s (exists)", trial_name)
                continue

            os.makedirs(out_dir, exist_ok=True)

            trial_cfg = ExperimentConfig(
                method=cfg.method,
                model=cfg.model,
                lora=cfg.lora,
                training=cfg.training,
                data=cfg.data,
                output=cfg.output,
            )
            trial_cfg.lora.rank = rank
            trial_cfg.training.learning_rate = lr

            try:
                _train_single(trial_cfg, train_ds, eval_ds, out_dir)
                results.append({"trial": trial_name, "rank": rank, "lr": lr, "status": "ok"})
            except Exception as e:
                logger.exception("Trial %s failed: %s", trial_name, e)
                results.append({"trial": trial_name, "rank": rank, "lr": lr, "status": str(e)})

    summary_path = os.path.join(output_root, "hparam_results.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    logger.info("Hparam search done. Summary: %s", summary_path)
