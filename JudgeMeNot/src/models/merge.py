"""Batch merge LoRA adapters into base model weights.

Produces full-weight model directories that can be loaded without PEFT,
used as the starting point for CLORA second-stage training or for faster
inference without adapter overhead .
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _ensure_tok_alignment(tok: AutoTokenizer, model: AutoModelForCausalLM):
    """Resize model embeddings if tokenizer has more tokens than the model."""
    model_vocab = model.get_input_embeddings().weight.shape[0]
    tok_vocab = len(tok)
    if tok_vocab != model_vocab:
        logger.info("Aligning embeddings: model=%d -> tokenizer=%d", model_vocab, tok_vocab)
        model.resize_token_embeddings(tok_vocab)


def merge_single_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: Optional[torch.dtype] = None,
    skip_existing: bool = True,
) -> Path:
    """
    Merge one LoRA adapter into its base model and save full weights.

    Parameters
    ----------
    base_model_id : str
        HuggingFace model ID or local path to the base model.
    adapter_path : str
        Path to the LoRA adapter directory (contains adapter_config.json).
    output_dir : str
        Where to save the merged model + tokenizer.
    torch_dtype : torch.dtype, optional
        Override dtype (default: bf16 if available, else fp16).
    skip_existing : bool
        If True, skip merging when output_dir already contains model weights.
    """
    from peft import PeftModel

    out = Path(output_dir)
    if skip_existing and out.exists() and (out / "config.json").exists():
        logger.info("Skipping %s (already exists)", out)
        return out

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading base: %s", base_model_id)
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    _ensure_tok_alignment(tok, model)

    logger.info("Attaching adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    logger.info("Saved merged model to %s", out)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


def batch_merge_adapters(
    base_model_id: str,
    adapters_root: str,
    output_root: str,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Merge all LoRA adapters found under adapters_root into the base model.

    Scans for directories containing adapter_config.json (directly or
    inside lora_adapter/ subdirectories).
    """
    adapters_root = Path(adapters_root)
    output_root = Path(output_root)
    merged = []

    for model_dir in sorted(adapters_root.iterdir()):
        if not model_dir.is_dir():
            continue

        adapter_dir = model_dir / "lora_adapter"
        if not adapter_dir.exists() or not (adapter_dir / "adapter_config.json").exists():
            if (model_dir / "adapter_config.json").exists():
                adapter_dir = model_dir
            else:
                logger.debug("No adapter in %s, skipping", model_dir.name)
                continue

        out_dir = output_root / model_dir.name
        try:
            result = merge_single_adapter(
                base_model_id=base_model_id,
                adapter_path=str(adapter_dir),
                output_dir=str(out_dir),
                skip_existing=skip_existing,
            )
            merged.append(result)
        except Exception as e:
            logger.exception("Failed to merge %s: %s", model_dir.name, e)

    logger.info("Merged %d / %d adapters", len(merged), len(list(adapters_root.iterdir())))
    return merged
