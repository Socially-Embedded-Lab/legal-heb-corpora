"""Unified model and tokenizer loading for all training and inference modes.

Consolidates 15+ duplicated load_model_and_tokenizer implementations into one
function that handles: Gemma NF4, DictaLM, AWQ, base-only, base+adapter,
local merged weights, and adapter vocab alignment .
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _is_gemma(name: str) -> bool:
    return "gemma" in name.lower()


def _make_bnb_config(quantize: str, compute_dtype: torch.dtype):
    """Build a BitsAndBytesConfig for NF4 quantization."""
    from transformers import BitsAndBytesConfig

    if quantize == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
        )
    return None


def _adapter_vocab_size(adapter_dir: Path) -> Optional[int]:
    """Infer the vocab size an adapter was trained with from saved weights."""
    saf_p = adapter_dir / "adapter_model.safetensors"
    bin_p = adapter_dir / "adapter_model.bin"
    weights = None
    try:
        if saf_p.exists():
            from safetensors.torch import load_file
            weights = load_file(str(saf_p))
        elif bin_p.exists():
            weights = torch.load(str(bin_p), map_location="cpu")
    except Exception:
        return None
    if not isinstance(weights, dict):
        return None
    for k, v in weights.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            if "embed_tokens" in k or "lm_head" in k:
                return int(v.shape[0])
    return None


def get_stop_token_ids(tokenizer) -> list[int]:
    """Collect EOS / end-of-turn token IDs for generation stopping."""
    candidates = ["<end_of_turn>", "<|eot_id|>", "<eos>", "</s>"]
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    for t in candidates:
        if t in vocab:
            ids.add(int(tokenizer.convert_tokens_to_ids(t)))
    if not ids and tokenizer.pad_token_id is not None:
        ids.add(int(tokenizer.pad_token_id))
    return list(ids)


def load_model(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    quantize: str = "nf4",
    device_map: str = "auto",
    is_trainable: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a causal-LM model and its tokenizer.

    Supports:
    - Base HF models (with optional NF4 quantization for Gemma)
    - Base + LoRA adapter (PEFT)
    - Local merged model directories
    - Automatic vocab alignment when adapter was trained with extra tokens

    Returns (tokenizer, model).
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # --- Determine the base checkpoint to load ---
    base_ckpt = model_name_or_path
    if adapter_path is not None:
        # When loading an adapter, read its base model from peft config
        from peft import PeftConfig
        try:
            peft_cfg = PeftConfig.from_pretrained(adapter_path)
            base_ckpt = peft_cfg.base_model_name_or_path
            logger.info("Adapter %s expects base: %s", adapter_path, base_ckpt)
        except Exception:
            logger.warning("Could not read PeftConfig from %s, using %s as base",
                           adapter_path, model_name_or_path)

    # --- Tokenizer ---
    tok_source = adapter_path if adapter_path else base_ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=trust_remote_code)

    if tokenizer.eos_token is None:
        base_tok = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=trust_remote_code)
        tokenizer.eos_token = base_tok.eos_token
        tokenizer.eos_token_id = base_tok.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Model kwargs ---
    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )

    # Quantization (only for Gemma-like or explicit nf4 request)
    should_quantize = quantize == "nf4" and (_is_gemma(base_ckpt) or _is_gemma(model_name_or_path))
    if should_quantize:
        model_kwargs["quantization_config"] = _make_bnb_config("nf4", dtype)

    # --- Load base model ---
    logger.info("Loading base model: %s (quantize=%s)", base_ckpt, quantize if should_quantize else "none")
    model = AutoModelForCausalLM.from_pretrained(base_ckpt, **model_kwargs)

    # --- Vocab alignment + adapter attachment ---
    if adapter_path is not None:
        adapter_dir = Path(adapter_path)
        adapter_vocab = _adapter_vocab_size(adapter_dir)
        base_vocab = model.get_input_embeddings().weight.shape[0]
        if adapter_vocab is not None and adapter_vocab != base_vocab:
            logger.info("Resizing base vocab %d -> adapter vocab %d", base_vocab, adapter_vocab)
            model.resize_token_embeddings(adapter_vocab)

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
        if not is_trainable:
            model.eval()
        logger.info("Attached adapter from %s", adapter_path)
    elif is_trainable:
        model.resize_token_embeddings(len(tokenizer))

    if not is_trainable:
        model.config.use_cache = True
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    return tokenizer, model


def find_adapter_dirs(models_root: Path) -> list[Path]:
    """Discover LoRA adapter directories under models_root.

    Checks (in order): lora_adapter/ subdir, latest checkpoint-*, direct
    adapter_config.json.
    """
    picked: list[Path] = []
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue

        la = model_dir / "lora_adapter"
        if la.is_dir() and (la / "adapter_config.json").exists():
            picked.append(la)
            continue

        latest = _latest_checkpoint(model_dir)
        if latest is not None:
            picked.append(latest)
            continue

        if (model_dir / "adapter_config.json").exists():
            picked.append(model_dir)
            continue

    # Deduplicate preserving order
    seen, unique = set(), []
    for p in picked:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _latest_checkpoint(model_dir: Path) -> Optional[Path]:
    ckpts = []
    for p in model_dir.glob("checkpoint-*"):
        if p.is_dir() and (p / "adapter_config.json").exists():
            m = re.match(r"checkpoint-(\d+)", p.name)
            num = int(m.group(1)) if m else -1
            ckpts.append((num, p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def split_items(sorted_items: list, n: int, which: int) -> list:
    """Split a sorted list into n roughly-equal parts and return part `which` (1-indexed)."""
    total = len(sorted_items)
    if total == 0:
        return []
    base = total // n
    rem = total % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    starts = [sum(sizes[:i]) for i in range(n)]
    idx = which - 1
    return sorted_items[starts[idx]: starts[idx] + sizes[idx]]
