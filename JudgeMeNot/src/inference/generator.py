"""Unified batch generation with resume, atomic writes, and OOM backoff.

Consolidates the generation/evaluation loops from eval_judges.py,
eval_them_all.py (CLORA), eval_them_all_rag.py, and evaluate_models.py .
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.models.loader import find_adapter_dirs, get_stop_token_ids, load_model, split_items
from src.inference.perplexity import batch_ppl_continuation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def atomic_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
#  Batch generation with OOM backoff
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> List[str]:
    """Tokenize prompts, generate, return only the new text per prompt."""
    batch_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    seq_len = batch_inputs["input_ids"].shape[1]

    stop_ids = get_stop_token_ids(tokenizer)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids if len(stop_ids) > 1 else stop_ids[0],
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))

    out = model.generate(**batch_inputs, **gen_kwargs)

    gens = []
    for i in range(out.shape[0]):
        gen_ids = out[i, seq_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gens.append(text.strip())
    return gens


def safe_generate_batch(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.7,
    try_mb: Optional[int] = None,
) -> List[str]:
    """Generate with OOM backoff: halve batch size on OOM, down to 1."""
    total = len(prompts)
    gens = [""] * total
    if total == 0:
        return gens

    mb = try_mb or total
    while mb >= 1:
        try:
            for start in range(0, total, mb):
                end = min(total, start + mb)
                sub = generate_batch(
                    tokenizer, model, prompts[start:end],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                gens[start:end] = sub
            return gens
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mb = mb // 2
            logger.warning("OOM during generate; retrying with microbatch=%d", max(mb, 1))

    # Per-sample fallback
    for i, p in enumerate(prompts):
        try:
            gens[i] = generate_batch(
                tokenizer, model, [p],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gens[i] = ""
            logger.error("Persistent OOM on row %d; leaving empty", i)
    return gens


# ---------------------------------------------------------------------------
#  Prompt builders
# ---------------------------------------------------------------------------

def build_chat_prompts(tokenizer, questions: List[str]) -> List[str]:
    """Build prompts using the tokenizer's chat template (IT/CLORA)."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]


def build_clm_prompts(texts: List[str], prompt_pct: float = 0.1) -> tuple[List[str], List[str]]:
    """Split CLM text into prompt (first N% words) and ground truth (rest)."""
    prompts, truths = [], []
    for text in texts:
        words = text.split()
        split_at = max(1, int(len(words) * prompt_pct))
        prompts.append(" ".join(words[:split_at]))
        truths.append(" ".join(words[split_at:]))
    return prompts, truths


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_adapters(
    models_root: str,
    data_root: str,
    out_root: str,
    base_model: str,
    method: str = "instruction",
    quantize: str = "nf4",
    batch_size: int = 4,
    cap_min: int = 1,
    cap_max: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.0,
    compute_ppl: bool = True,
    ppl_microbatch: int = 4,
    split_n: Optional[int] = None,
    split_which: Optional[int] = None,
    prompt_pct: float = 0.1,
    rag_manager=None,
    rag_k: int = 3,
    rag_exact_threshold: float = 0.95,
    rag_context_modes: Optional[List[str]] = None,
    attn_implementation: str = "eager",
):
    """
    Evaluate all adapters under models_root against all test CSVs under data_root.

    Supports instruction (chat), CLM (continuation), and RAG-enhanced evaluation.
    Produces one CSV per (adapter, test_dataset) pair with columns:
      question, ground_truth, generated, [perplexity], [rag_context, rag_sources]
    """
    models_root = Path(models_root)
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    adapter_dirs = find_adapter_dirs(models_root)
    if not adapter_dirs:
        logger.error("No adapters found under %s", models_root)
        return

    if split_n and split_which:
        adapter_dirs = split_items(adapter_dirs, split_n, split_which)

    logger.info("Found %d adapter(s) to evaluate", len(adapter_dirs))

    # Discover judge CSVs (flat layout: one CSV per judge with a 'split' column)
    all_csvs = sorted(
        p for p in data_root.glob("*.csv") if p.stem != "leakage_report"
    )
    if not all_csvs:
        logger.error("No judge CSVs found under %s", data_root)
        return

    context_modes = rag_context_modes or (["no_rag"] if rag_manager is None else ["own_context", "others_context"])

    for adapter_dir in tqdm(adapter_dirs, desc="Models", unit="model"):
        model_name = adapter_dir.parent.name

        try:
            tokenizer, model = load_model(
                base_model,
                adapter_path=str(adapter_dir),
                quantize=quantize,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", model_name, e)
            torch.cuda.empty_cache()
            gc.collect()
            continue

        for ctx_mode in context_modes:
            ctx_out = out_root / ctx_mode if rag_manager else out_root

            for csv_path in tqdm(all_csvs, desc=f"{model_name}:{ctx_mode}", leave=False):
                _evaluate_single_csv(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name,
                    csv_path=csv_path,
                    out_dir=ctx_out,
                    method=method,
                    batch_size=batch_size,
                    cap_min=cap_min,
                    cap_max=cap_max,
                    do_sample=do_sample,
                    temperature=temperature,
                    compute_ppl=compute_ppl,
                    ppl_microbatch=ppl_microbatch,
                    prompt_pct=prompt_pct,
                    rag_manager=rag_manager,
                    rag_k=rag_k,
                    rag_exact_threshold=rag_exact_threshold,
                    rag_context_name=(
                        model_name if ctx_mode == "own_context"
                        else csv_path.stem if ctx_mode == "others_context"
                        else None
                    ),
                )

        del tokenizer, model
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All evaluations complete. Results in %s", out_root)


def _evaluate_single_csv(
    tokenizer,
    model,
    model_name: str,
    csv_path: Path,
    out_dir: Path,
    method: str,
    batch_size: int,
    cap_min: int,
    cap_max: int,
    do_sample: bool,
    temperature: float,
    compute_ppl: bool,
    ppl_microbatch: int,
    prompt_pct: float,
    rag_manager,
    rag_k: int,
    rag_exact_threshold: float,
    rag_context_name: Optional[str],
):
    """Evaluate one adapter on one test CSV (flat layout with ``split`` column)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Skipping %s (read error: %s)", csv_path, e)
        return

    if "split" in df.columns:
        df = df[df["split"] == "test"].reset_index(drop=True)
    if df.empty:
        logger.warning("Skipping %s (no test rows)", csv_path)
        return

    # Resolve columns
    q_cols = ["question", "Question", "prompt", "input"]
    a_cols = ["answer", "ground_truth", "GroundTruth", "target", "reference"]
    q_col = next((c for c in q_cols if c in df.columns), None)
    a_col = next((c for c in a_cols if c in df.columns), None)
    if q_col is None or a_col is None:
        logger.warning("Skipping %s (missing columns)", csv_path)
        return

    dataset_name = csv_path.stem
    suffix = "_rag_data" if rag_manager else "_data"
    out_name = f"{safe_name(model_name)}_model_vs_{safe_name(dataset_name)}{suffix}.csv"
    out_path = out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = df[q_col].astype(str).tolist()
    truths = df[a_col].astype(str).tolist()
    N = len(questions)

    # Resume logic
    generated = [""] * N
    perplexities = [None] * N
    rag_contexts_col = [""] * N
    rag_sources_col = [""] * N

    if out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            if len(prev) == N and "generated" in prev.columns:
                generated = prev["generated"].astype(str).fillna("").tolist()
                if "perplexity" in prev.columns:
                    perplexities = prev["perplexity"].tolist()
                if "rag_context" in prev.columns:
                    rag_contexts_col = prev["rag_context"].astype(str).fillna("").tolist()
                if "rag_sources" in prev.columns:
                    rag_sources_col = prev["rag_sources"].astype(str).fillna("").tolist()
        except Exception:
            pass

    def _needs_work(i):
        if not generated[i]:
            return True
        if compute_ppl and (perplexities[i] is None or (isinstance(perplexities[i], float) and pd.isna(perplexities[i]))):
            return True
        return False

    todo_idx = [i for i in range(N) if _needs_work(i)]
    if not todo_idx:
        logger.info("Complete: %s", out_path)
        return

    # Token cap from ground truth
    gt_lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in truths]
    max_gt = max(gt_lens) if gt_lens else 1
    cap = max(cap_min, min(cap_max, int(math.ceil(max_gt * 1.15))))

    for j in range(0, len(todo_idx), batch_size):
        batch_idx = todo_idx[j: j + batch_size]
        batch_q = [questions[i] for i in batch_idx]
        batch_gt = [truths[i] for i in batch_idx]

        # Build prompts
        if method == "clm":
            prompts = batch_q  # already split upstream
        elif rag_manager and rag_context_name:
            from src.data.rag_store import build_rag_prompts
            batch_ctxs = []
            batch_srcs = []
            for q in batch_q:
                ctxs, srcs = rag_manager.retrieve(rag_context_name, q, k=rag_k, exact_match_threshold=rag_exact_threshold)
                batch_ctxs.append(ctxs)
                batch_srcs.append(srcs)
            prompts = build_rag_prompts(tokenizer, batch_q, batch_ctxs)
        else:
            prompts = build_chat_prompts(tokenizer, batch_q)

        # Generate
        gens = safe_generate_batch(
            tokenizer, model, prompts,
            max_new_tokens=cap,
            do_sample=do_sample,
            temperature=temperature,
            try_mb=batch_size,
        )

        # PPL
        ppls = [None] * len(batch_idx)
        if compute_ppl:
            torch.cuda.empty_cache()
            ppls = batch_ppl_continuation(tokenizer, model, prompts, batch_gt, microbatch=ppl_microbatch)

        # Store
        for k_idx, (g, ppl) in enumerate(zip(gens, ppls)):
            i = batch_idx[k_idx]
            if g:
                generated[i] = g
            perplexities[i] = float(ppl) if ppl is not None else float("nan")
            if rag_manager and rag_context_name:
                rag_contexts_col[i] = "\n---\n".join(batch_ctxs[k_idx]) if batch_ctxs[k_idx] else ""
                rag_sources_col[i] = json.dumps(batch_srcs[k_idx], ensure_ascii=False) if batch_srcs[k_idx] else ""

        # Incremental save
        save_dict = {"question": questions, "ground_truth": truths, "generated": generated}
        if compute_ppl:
            save_dict["perplexity"] = perplexities
        if rag_manager:
            save_dict["rag_context"] = rag_contexts_col
            save_dict["rag_sources"] = rag_sources_col
        atomic_write_csv(pd.DataFrame(save_dict), out_path)

    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
#  Base-model-only evaluation (no adapter)
# ---------------------------------------------------------------------------

def evaluate_base_model(
    model_name: str,
    data_root: str,
    out_root: str,
    method: str = "instruction",
    quantize: str = "nf4",
    batch_size: int = 4,
    cap_min: int = 1,
    cap_max: int = 1024,
    compute_ppl: bool = True,
    ppl_microbatch: int = 4,
    attn_implementation: str = "eager",
):
    """Evaluate a base HF model (no adapter) across all test CSVs."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(
        model_name, quantize=quantize, attn_implementation=attn_implementation,
    )

    all_csvs = sorted(
        p for p in data_root.glob("*.csv") if p.stem != "leakage_report"
    )
    mn = safe_name(model_name.split("/")[-1])

    for csv_path in tqdm(all_csvs, desc=f"Base:{mn}"):
        _evaluate_single_csv(
            tokenizer=tokenizer,
            model=model,
            model_name=mn,
            csv_path=csv_path,
            out_dir=out_root,
            method=method,
            batch_size=batch_size,
            cap_min=cap_min,
            cap_max=cap_max,
            do_sample=False,
            temperature=0.0,
            compute_ppl=compute_ppl,
            ppl_microbatch=ppl_microbatch,
            prompt_pct=0.1,
            rag_manager=None,
            rag_k=0,
            rag_exact_threshold=0.95,
            rag_context_name=None,
        )

    del tokenizer, model
    torch.cuda.empty_cache()
    gc.collect()
