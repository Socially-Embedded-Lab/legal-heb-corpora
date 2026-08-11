# JudgeMeNot

This repository contains the replication code for the ACL 2026 paper "JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew". Given a corpus of rulings written by individual judges, we train judge-specific adapters so the model learns each judge's linguistic style, reasoning patterns, and domain terminology -- then generates answers that sound like that particular judge wrote them.

## Status: missing module

**`src/data/` is absent from this repository.** It was never included in either
upstream commit (it is not a `.gitignore` artifact — that file is empty). Five modules
import from it:

| Import | Site | When it fails |
|--------|------|---------------|
| `src.data.datasets` | `src/training/trainer.py:30` | module level — on import |
| `src.data.datasets` | `src/analysis/ablation.py:18,20` | module level — on import |
| `src.data.rag_store` | `run.py:62` (`cmd_eval`) | when RAG eval runs |
| `src.data.rag_store` | `run.py:141` (`cmd_rag_index`) | when called |
| `src.data.rag_store` | `src/inference/generator.py:365` | when RAG generation runs |

The missing module must provide:

* `datasets.py` — `csvs_to_messages_dataset`, `discover_judges`, `load_clm_dataset`,
  `load_judge_csvs`, `create_data_subset`
* `rag_store.py` — `RAGManager`, `build_rag_index`, `build_rag_prompts`

**Working (7 of 12 commands):** `eval` (without RAG), `metrics`, `merge`, `baselines`,
`bootstrap`, `clf-data`, `clf-train`, `clf-eval`.
**Broken (5 of 12):** `train`, `hparam`, `ablation`, `rag-index`, `eval --rag`.

This has not been reconstructed here — the dataset loading and retrieval logic is
methodologically load-bearing for the paper's results, so it should come from the
authors rather than be reimplemented. Request it from the upstream repository.

---

## Methods

The project implements four complementary approaches, all built on top of **Gemma** (1B / 4B) and **DictaLM** base models using LoRA (Low-Rank Adaptation):

| Method | Key idea | Data format |
|---|---|---|
| **Causal LM (CLM)** | Continue writing in the judge's voice given a text prefix | Plain text chunks from rulings |
| **Instruction Tuning (IT)** | Answer legal questions the way the judge would | Question / answer CSV pairs |
| **CLORA** | Instruction-tune on a base that was already merged with a CLM adapter | Same as IT, but on pre-merged weights |
| **RAG** | Retrieve similar past Q&A pairs as context before generating | IT data + FAISS vector index |

Each method trains a separate LoRA adapter per judge, producing 29 adapters per method.

Base models (Gemma, DictaLM) can also be evaluated directly without any LoRA adapter via `--base_only`.

## Evaluation

### Text metrics

Generated answers are compared against ground-truth using:

- **BLEU** (sacrebleu) and **ROUGE** (rouge-1/2/L with Hebrew UDPipe tokenization)
- **BERTScore** (DictaBERT as the backbone)
- **Perplexity** over the ground-truth continuation
- **POS distribution divergence** (JSD via DictaBERT-parse)

### Classifier-based evaluation (discernment test)

Per-judge **DictaBERT binary classifiers** are trained to distinguish a judge's real text from other judges' text. Generated text is then tested against these classifiers: if the classifier cannot tell generated text apart from real text (low accuracy / high failure rate), the generation successfully mimics the judge's style.

### Statistical significance

- **Bootstrap confidence intervals** for all metrics
- **Bootstrap delta analysis** (own-judge score minus cross-judge average)
- **Wilcoxon signed-rank tests** for pairwise method comparisons

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+ and a CUDA-capable GPU for training and inference.

> **Secrets:** Do not commit Hugging Face or other API tokens. Use `huggingface-cli
> login` or set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in your environment.

## Usage

Everything runs through a single CLI entry point:

```bash
python run.py <command> [options]
```

### Training

```bash
# Instruction Tuning -- trains one LoRA adapter per judge
python run.py train --config configs/experiments/it_gemma4b.yaml

# Causal LM -- trains on text-chunk datasets
python run.py train --config configs/experiments/clm_gemma4b.yaml

# CLORA -- instruction-tune on pre-merged CLM bases
python run.py train --config configs/experiments/clora_gemma4b.yaml
```

### Evaluation

```bash
# Generate answers from all trained adapters
python run.py eval --config configs/experiments/eval_standard.yaml \
    --models_root outputs/adapters/it_gemma4b \
    --data_root data/qa

# Evaluate base model without adapters (e.g. DictaLM, Gemma)
python run.py eval --config configs/experiments/eval_standard.yaml \
    --data_root data/qa --base_only

# Evaluate a specific base model
python run.py eval --config configs/experiments/eval_standard.yaml \
    --data_root data/qa --base_only \
    --base_model dicta-il/dictalm2.0-instruct
```

### Metrics

```bash
# Add text metrics to all evaluation CSVs
python run.py metrics --input_dir outputs/eval --metrics bleu,rouge --recursive

# Add BERTScore
python run.py metrics --input_dir outputs/eval --metrics bertscore

# Add style (POS JSD divergence)
python run.py metrics --input_dir outputs/eval --metrics style
```

### RAG

```bash
# 1. Build per-judge FAISS vector indices
python run.py rag-index \
    --data_root data/qa \
    --cache_dir outputs/rag_cache

# 2. Evaluate with retrieved context
python run.py eval --config configs/experiments/rag.yaml \
    --rag_cache outputs/rag_cache \
    --models_root outputs/adapters/it_gemma4b \
    --data_root data/qa
```

### Merge LoRA into base weights

```bash
python run.py merge \
    --adapters_root outputs/adapters/clm_gemma4b \
    --base_model google/gemma-3-4b-it \
    --output outputs/merged
```

CLORA then reads these merged bases via `${paths.merged_root}`, so the merge output
directory and `merged_root` in `configs/paths.yaml` must agree — both are `outputs/merged`.

### Classifier (discernment evaluation)

```bash
# 1. Prepare per-judge binary datasets (from local QA or chunk CSVs)
python run.py clf-data --data_root data/qa \
    --output_dir outputs/classifier_datasets

# 2. Train DictaBERT classifiers
python run.py clf-train \
    --dataset_dir outputs/classifier_datasets \
    --output_dir outputs/classifiers

# 3. Evaluate: can generated text fool the classifiers?
python run.py clf-eval \
    --classifier_dir outputs/classifiers \
    --eval_dir outputs/eval/it_gemma4b \
    --dataset_dir outputs/classifier_datasets \
    --output_dir outputs/fooling/it_gemma4b \
    --method_name it-4b --stats
```

### Ablation studies

```bash
# Data-size ablation (25%, 50%, 75%, 100% of training data)
python run.py ablation --config configs/experiments/ablation_data_size.yaml \
    --judge Judge_A --type data_size

# LoRA rank ablation (ranks 2, 4, 8, 16, 32)
python run.py ablation --config configs/experiments/ablation_lora_rank.yaml \
    --judge Judge_A --type lora_rank
```

### Statistical analysis

```bash
# Bootstrap confidence intervals
python run.py bootstrap --input_dir outputs/eval/it --output_dir outputs/analysis

# Bootstrap delta analysis (own-judge vs cross-judge comparison)
python run.py bootstrap --input_dir outputs/eval/it --output_dir outputs/analysis --delta

# Wilcoxon signed-rank baseline comparisons
python run.py baselines --input_dir outputs/analysis/jsd_pos --lower_is_better
```

### Hyperparameter search

```bash
python run.py hparam --config configs/experiments/it_gemma4b.yaml \
    --train_csv data/qa/Judge_A.csv \
    --ranks 4,8,16,32 --lrs 1e-4,2e-4,5e-4
```

## Configuration

Experiments are driven by YAML configs in `configs/`:

```
configs/
  paths.yaml                    # Default paths (data root, output root)
  models/
    gemma_1b.yaml               # Gemma 1B model settings
    gemma_4b.yaml               # Gemma 4B model settings
    dictalm.yaml                # DictaLM model settings
  experiments/
    clm_gemma4b.yaml            # CLM training config
    it_gemma4b.yaml             # Instruction Tuning config
    clora_gemma4b.yaml          # CLORA config
    eval_standard.yaml          # Standard evaluation config
    rag.yaml                    # RAG evaluation config
    ablation_data_size.yaml     # Data-size ablation config
    ablation_lora_rank.yaml     # LoRA-rank ablation config
```

Paths use `${paths.key}` placeholders that resolve against `configs/paths.yaml`:

```yaml
# configs/paths.yaml
local:
  data_root: ./data/qa
  output_root: ./outputs
  rag_cache: ./outputs/rag_cache
  merged_root: ./outputs/merged
```

```yaml
# referenced from an experiment config as:
data:
  merged_root: ${paths.merged_root}
```

Resolution is handled by `src/config.py`, which locates `configs/paths.yaml` relative
to the repo root, so commands work from any working directory. Only the `local`
environment block is read.

## Data layout

The code expects data to be placed under the `data` sub-directory. Judge names are masked as Judge_A through Judge_AC. Each judge has a single flat CSV with a `split` column indicating the data fold:

```
data/
  qa/
    Judge_A.csv    # columns: question, answer, split
    Judge_B.csv
    ...
    Judge_AC.csv   # 29 judges total
  chunks/
    Judge_A.csv    # columns: text, split
    Judge_B.csv
    ...
    Judge_AC.csv
```

## Project structure

```
run.py                          # Single CLI entry point (12 subcommands)
requirements.txt
configs/                        # All YAML configuration
src/
  config.py                     # Dataclasses + YAML loader with path resolution
  models/
    loader.py                   # Unified model/tokenizer loading (NF4, adapters, merged)
    merge.py                    # Merge LoRA adapters into base weights
  data/                         # !! NOT IN THIS REPO -- see "Status" above
    datasets.py                 # Dataset loading for CLM / IT / CLORA
    rag_store.py                # FAISS index creation + RAGManager retrieval
  training/
    trainer.py                  # Unified LoRA SFT training + hparam search
  inference/
    generator.py                # Batch generation with resume + OOM backoff
    perplexity.py               # Continuation perplexity (prompt-masked)
  metrics/
    text_metrics.py             # BLEU + ROUGE (Hebrew UDPipe tokenizer)
    bertscore.py                # BERTScore with DictaBERT
    style.py                    # POS distribution JSD divergence
    bootstrap.py                # Bootstrap CI + delta significance analysis
  classifier/
    data.py                     # Per-judge binary dataset prep (balance + split)
    train.py                    # DictaBERT classifier training
    evaluate.py                 # Fooling evaluation + statistical tests
  analysis/
    baselines.py                # Wilcoxon signed-rank tests
    ablation.py                 # Data-size and LoRA-rank ablation orchestration
    plots.py                    # Forest, heatmap, CI, ablation, cross-method plots
```
