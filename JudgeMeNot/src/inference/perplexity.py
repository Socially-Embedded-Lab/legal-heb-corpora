"""Masked-continuation perplexity computation.

Computes per-example perplexity by masking the prompt tokens in the loss
and scoring only the continuation (groundtruth answer).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def batch_ppl_continuation(
    tokenizer,
    model,
    prompts: List[str],
    continuations: List[str],
    microbatch: int = 4,
) -> List[Optional[float]]:
    """Compute perplexity over continuation text given prompts.

    For each (prompt, continuation) pair, tokenizes prompt+continuation,
    masks prompt tokens in labels, and computes cross-entropy only on the
    continuation tokens.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel
    prompts : list of str
        The prompt strings (context the model conditions on).
    continuations : list of str
        The ground-truth continuation strings to score.
    microbatch : int
        Sub-batch size to avoid OOM.

    Returns
    -------
    list of float or None
        Per-example perplexity values. None for rows with no scoreable tokens.
    """
    if not prompts:
        return []

    device = model.device
    B = len(prompts)
    mb = max(1, microbatch)
    out_ppls: List[Optional[float]] = [None] * B

    for start in range(0, B, mb):
        end = min(B, start + mb)
        p_slice = prompts[start:end]
        c_slice = continuations[start:end]

        # Tokenize prompts alone to get their lengths
        enc_prompts = tokenizer(p_slice, return_tensors="pt", padding=True, truncation=True)
        prompt_lens = (enc_prompts["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

        # Tokenize full text (prompt + continuation)
        full_texts = [p + c for p, c in zip(p_slice, c_slice)]
        enc_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc_full["input_ids"].to(device)
        attention_mask = enc_full["attention_mask"].to(device)
        labels = input_ids.clone()

        # Mask prompt tokens and padding
        for i in range(labels.size(0)):
            labels[i, : int(prompt_lens[i].item())] = -100
        labels[attention_mask == 0] = -100

        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits[:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            mask = target != -100

            V = logits.size(-1)
            loss_flat = F.cross_entropy(
                logits.view(-1, V),
                target.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(input_ids.size(0), -1)

            token_sums = (loss_flat * mask).sum(dim=1)
            token_counts = mask.sum(dim=1).clamp_min(1)
            mean_losses = token_sums / token_counts

            for i in range(input_ids.size(0)):
                if int(token_counts[i].item()) == 0:
                    out_ppls[start + i] = None
                else:
                    out_ppls[start + i] = float(torch.exp(mean_losses[i]).item())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM during PPL computation at batch %d-%d", start, end)
        finally:
            del enc_prompts, enc_full, input_ids, attention_mask, labels
            try:
                del logits, target, mask, loss_flat, token_sums, token_counts, mean_losses
            except NameError:
                pass
            torch.cuda.empty_cache()

    return out_ppls
