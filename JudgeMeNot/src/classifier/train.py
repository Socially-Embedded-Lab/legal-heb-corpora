"""Train per-judge DictaBERT binary classifiers.

Each judge gets a binary sequence classifier (DictaBERT by default):
label 1 = text authored by this judge, label 0 = text from other judges.
Training uses HuggingFace Trainer with early stopping on validation F1 .
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "dicta-il/dictabert"


def _compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for Trainer."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def train_single_classifier(
    judge: str,
    dataset_dir: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    patience: int = 3,
    seed: int = 42,
) -> dict:
    """Train a binary classifier for one judge.

    Reads {judge}_train.json / _validation.json / _test.json from dataset_dir.
    Saves model + tokenizer to output_dir/{judge}/.
    Returns dict with test metrics.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    ds_root = Path(dataset_dir) / judge
    out_root = Path(output_dir) / judge
    out_root.mkdir(parents=True, exist_ok=True)

    def _load_split(name: str) -> Dataset:
        path = ds_root / f"{judge}_{name}.json"
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return Dataset.from_dict({
            "text": [it["text"] for it in items],
            "label": [it["label"] for it in items],
        })

    train_ds = _load_split("train")
    val_ds = _load_split("validation")
    test_ds = _load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(_tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(out_root / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        seed=seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()
    test_results = trainer.evaluate(test_ds)

    model.save_pretrained(str(out_root))
    tokenizer.save_pretrained(str(out_root))

    results = {
        "judge": judge,
        "model": model_name,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        **{k.replace("eval_", "test_"): v for k, v in test_results.items()},
    }

    with open(out_root / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def train_all_classifiers(
    dataset_dir: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    patience: int = 3,
    judges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Train classifiers for all judges found in dataset_dir.

    Returns summary DataFrame with test metrics per judge.
    """
    ds_root = Path(dataset_dir)
    if judges is None:
        judges = sorted(
            d.name for d in ds_root.iterdir()
            if d.is_dir() and (d / f"{d.name}_train.json").exists()
        )

    all_results = []
    for judge in tqdm(judges, desc="Training classifiers"):
        logger.info("Training classifier for %s", judge)
        try:
            res = train_single_classifier(
                judge=judge,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_length=max_length,
                patience=patience,
            )
            all_results.append(res)
        except Exception as e:
            logger.error("Failed training for %s: %s", judge, e)
            all_results.append({"judge": judge, "error": str(e)})

    summary = pd.DataFrame(all_results)
    out_path = Path(output_dir) / "training_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Training summary saved to %s", out_path)
    return summary
