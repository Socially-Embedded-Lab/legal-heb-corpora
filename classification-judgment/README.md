# Classification Judgment — Hebrew Legal Text Classifier

Automatic classification of paragraphs from Israeli court verdicts into six semantic
categories, comparing fine-tuned Hebrew BERT models against GPT-4.1-mini few-shot
strategies.

---

## Overview

Israeli court judgments follow a predictable structure but vary significantly in
formatting and language. This project builds and evaluates a pipeline that:

1. Extracts paragraphs from Hebrew DOCX verdict files
2. Identifies section boundaries using document formatting cues (bold, Miriam font)
3. Classifies each paragraph into one of six legal categories
4. Compares fine-tuned BERT models against GPT few-shot classification strategies

Three corpora are covered: **drug offenses**, **weapons offenses**, and a
**no-titles** set of verdicts that lack section headers entirely.

---

## Classification Categories

The `label` column stores the Hebrew category name; [`01_finetune_bert.ipynb`](notebooks/01_finetune_bert.ipynb) maps it to an
integer id via `label2id`.

| ID | Hebrew | Description | Paragraphs |
|----|--------|-------------|-----------:|
| 0 | עובדות המקרה | Case Facts — background and factual descriptions | 549 |
| 1 | ראיות לעונש | Sentencing Evidence — evidence relevant to sentencing | 114 |
| 2 | טיעוני הצדדים | Parties' Arguments — claims by prosecution and defense | 647 |
| 3 | תסקיר שירות מבחן | Probation Service Report | 277 |
| 4 | דיון והכרעה | Discussion & Ruling — judicial analysis and decision | 2,528 |
| 5 | מידע אחר | Other Information — miscellaneous content | 154 |

The dataset is heavily imbalanced — *Discussion & Ruling* alone is 59% of all
paragraphs, while *Sentencing Evidence* is under 3%. Macro-averaged metrics are
therefore substantially lower than accuracy, and are the more informative figure.

---

## Project Structure

```
classification-judgment/
├── config/
│   └── config.yaml                      # Corpora, paths, OpenAI API key
├── data/
│   ├── raw/                             # Source DOCX verdicts
│   │   ├── drugs/                       #   30 files
│   │   ├── weapon/                      #   30 files
│   │   └── no_titles/                   #   27 files (no section headers)
│   ├── interim/                         # Extracted paragraph CSVs, one per verdict
│   │   ├── drugs/  weapon/              #   30 files each
│   │   └── no_titles/                   #   87 files
│   ├── paragraphs/                      # Extracted paragraphs for a much larger
│   │   ├── drugs/                       #   corpus — 2,936 verdicts
│   │   └── weapon/                      #   1,624 verdicts (sources not in repo)
│   └── labeled/                         # Ground truth
│       ├── drugs_combined.csv           #   1,845 paragraphs
│       ├── weapon_combined.csv          #   1,488 paragraphs
│       ├── no_titles_combined.csv       #     936 paragraphs
│       ├── all_combined.csv             #   4,269 — adds a `type` column
│       ├── all_combined_sorted.csv      #   same rows, sorted, no `type`
│       └── *.xlsx                       #   Excel mirrors of the per-corpus CSVs
├── notebooks/                           # Numbered in pipeline order
│   ├── 01_finetune_bert.ipynb           # Fine-tune AlephBERT & DictaBERT
│   ├── 02_gpt_titled.ipynb              # GPT — drugs/weapon corpora
│   ├── 03_gpt_whole_verdict.ipynb       # GPT — incl. whole-verdict pass
│   ├── 04_gpt_headerless.ipynb          # GPT — no-titles corpus
│   └── 05_header_detection.ipynb        # Header detection & section analysis
├── src/
│   └── extract_paragraphs.py            # DOCX → CSV paragraph extractor
├── results/
│   ├── fine_tuning/
│   │   ├── finetuned_alephbert/best/    # Tokenizer + config only — no weights
│   │   ├── finetuned_dicta/best/        # Tokenizer + config only — no weights
│   │   ├── test_metrics_summary.csv     # Test-set metrics for both models
│   │   ├── test_predictions.csv         # Per-paragraph test predictions
│   │   ├── validation_predictions.csv   # Per-paragraph validation predictions
│   │   └── all_predictions.csv          # All splits combined
│   ├── drugs/  ·  weapon/               # per-strategy dirs: only_title, only_text,
│   │                                    #   text_and_title, whole_case
│   │                                    # + all_results.csv (one row per paragraph,
│   │                                    #   one column per strategy)
│   │                                    # + gpt_headers_detection_results.csv
│   └── no_titles/                       # one_last_class, last_text_and_classes,
│                                        #   whole_case + all_results.csv
│                                        #   + all_statistics_summary.csv
└── docs/
    └── report.docx                      # Written project report
```

---

## Installation

```bash
pip install pandas numpy transformers torch scikit-learn datasets
pip install python-docx pyyaml openai
pip install matplotlib seaborn arabic-reshaper python-bidi
pip install pywin32   # Windows only — for .doc → .docx conversion
```

> Requires Python 3.8+ and a CUDA-capable GPU for fine-tuning (recommended).

---

## Configuration

Edit [`config/config.yaml`](config/config.yaml) before running:

```yaml
TYPES: ['drugs', 'weapon']               # Corpora to process (also: no_titles)
DOCX_PATH: data/raw/{type}               # Input DOCX verdicts
CSV_PATH: data/interim/{type}            # Per-verdict paragraph CSVs
PARAGRAPHS_PATH: data/paragraphs/{type}  # Intermediate per-paragraph dumps
GT_PATH: data/labeled                    # Ground-truth labeled datasets
RESULTS_PATH: results                    # Model and GPT outputs
OPENAI_API_KEY:                          # Place your API key here
```

> **`config.yaml` is tracked in git.** It ships with an empty `OPENAI_API_KEY`
> placeholder — but once you paste a real key in, `git add` will happily commit it.
> Before adding your key, untrack the file locally:
>
> ```bash
> git update-index --skip-worktree config/config.yaml
> ```

> The corpus keys `drugs` / `weapon` / `no_titles` are data values as well as
> directory names — they appear in the `type` column of
> `data/labeled/all_combined.csv`. Renaming them desyncs the code from the data.

### Working directories

- **Notebooks** run from `notebooks/`, resolving everything against `..`.
- **`extract_paragraphs.py`** resolves paths against the repo root via `__file__`,
  so it runs from any working directory.

Upstream, paths were a mix of absolute `C:\Users\...` and `/home/tak/...` literals
that only worked on the original authors' machines; these have been rewritten to be
repo-relative. Preserved notebook *outputs* still show the original absolute paths —
that is a historical record of the run, not a live path.

---

## Pipeline

### Step 1 — Extract Paragraphs from DOCX

```bash
python src/extract_paragraphs.py
```

Reads verdict files from `data/raw/{type}/`, detects section boundaries using
bold/Miriam-font formatting, strips metadata headers, and writes one CSV per verdict
to `data/interim/{type}/`.

Each output row contains:

| Column | Description |
|--------|-------------|
| `verdict` | Source file name |
| `text` | Paragraph text |
| `part` | Detected section title (`nothing` when none was found) |

**Notes:**
- Handles Hebrew complex-script bold via XML-level `w:bCs` inspection — plain `w:b`
  is unreliable for right-to-left runs
- Automatically converts `.doc` to `.docx` on Windows (requires pywin32); the import
  is guarded, so the rest of the script runs fine on macOS/Linux
- Deduplicates on `text` and skips quoted-only content

### Step 2 — Ground Truth Preparation

Manually labeled CSVs live in `data/labeled/`. Pre-labeled files are included for all three
corpora, totaling **4,269 labeled paragraphs**.

| Corpus | Paragraphs |
|--------|-----------:|
| Drugs | 1,845 |
| Weapons | 1,488 |
| No titles | 936 |
| **Total** | **4,269** |

### Step 3 — Fine-tune BERT Models

Run [`notebooks/01_finetune_bert.ipynb`](notebooks/01_finetune_bert.ipynb).

- **Models:** [AlephBERT](https://huggingface.co/onlplab/alephbert-base) (general
  Hebrew) and [DictaBERT](https://huggingface.co/dicta-il/dictabert) (Hebrew,
  including legal text)
- **Split:** 70% train / 15% validation / 15% test — implemented as two successive
  `train_test_split` calls (`test_size=0.15`, then `test_size=0.176` on the remainder)
- **Architecture:** `BertForSequenceClassification`, 6 output classes
- **Hyperparameters:** 5 epochs, learning rate `2e-5`, `max_length=256`
- **Output:** `results/fine_tuning/finetuned_*/best/` — **tokenizer and config only.**
  The trained weights were never committed (no `.safetensors`/`.bin` in the repo), so
  these directories cannot be loaded as a working classifier. Re-run this notebook to
  reproduce the weights; the metrics below come from the original run.

**Test-set results** (from `results/fine_tuning/test_metrics_summary.csv`):

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|---------:|------------------:|---------------:|-----------:|
| AlephBERT | 91.11% | 88.31% | 86.28% | 87.23% |
| DictaBERT | **91.89%** | 87.83% | 85.97% | 86.84% |

DictaBERT edges ahead on accuracy; AlephBERT is marginally better on macro-averaged
metrics, meaning it handles the rare classes slightly more evenly. The two are
effectively tied.

### Step 4 — GPT-based Classification

Few-shot classification with **`gpt-4.1-mini`**, using Hebrew system prompts. Which
strategies are available depends on the corpus.

**Drugs and weapons** — via [`02_gpt_titled.ipynb`](notebooks/02_gpt_titled.ipynb) and
[`03_gpt_whole_verdict.ipynb`](notebooks/03_gpt_whole_verdict.ipynb):

| Strategy | Output dir | Input given to the model |
|----------|-----------|--------------------------|
| Title only | `only_title/` | Detected section header alone |
| Text only | `only_text/` | Paragraph text alone |
| Title + text | `text_and_title/` | Both |
| Whole verdict | `whole_case/` | Entire document classified in one pass |

**No-titles corpus** — via [`04_gpt_headerless.ipynb`](notebooks/04_gpt_headerless.ipynb),
which targets
`TYPES = ['no_titles']`. Since these verdicts have no headers, the strategies instead
lean on previously assigned labels as context:

| Strategy | Output dir | Input given to the model |
|----------|-----------|--------------------------|
| One last class | `one_last_class/` | Paragraph + the preceding paragraph's label |
| Last text + classes | `last_text_and_classes/` | Paragraph + preceding text and labels |
| Whole verdict | `whole_case/` | Entire document classified in one pass |

Per-strategy CSVs are written to `results/{type}/{strategy}/`, and merged into
`results/{type}/all_results.csv` with one column per strategy.

**Accuracy by strategy** (computed from `all_results.csv` against the `label` column):

| Corpus | Title only | Text only | Title + text | Whole verdict |
|--------|-----------:|----------:|-------------:|--------------:|
| Drugs (n=1,845) | 98.75% | 93.71% | **99.02%** | 94.42% |
| Weapons (n=1,488) | 97.78% | 94.56% | **98.19%** | 95.50% |

The section header turns out to be a very strong signal on its own — title-only beats
text-only by ~4-5 points, and combining the two is best. This also means the headline
numbers are not directly comparable to the BERT results above, which classify from
text alone.

For the no-titles corpus, `results/no_titles/all_statistics_summary.csv` reports
**87.18%** (one last class) and **92.95%** (last text + classes) over the 936 labeled
paragraphs.

### Step 5 — Header Detection & Analysis

Run [`notebooks/05_header_detection.ipynb`](notebooks/05_header_detection.ipynb) to detect the presence of the three main verdict sections (facts / arguments /
ruling) across the corpus and generate distribution statistics. Output lands in
`results/{type}/gpt_headers_detection_results.csv`.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (macro & micro)
- Per-class classification report
- Confusion matrices

Given the class imbalance noted above, prefer macro-F1 when comparing models.

---

## Reproducibility Notes

- `results/no_titles/all_results.csv` contains all **4,269** rows (the full combined
  dataset), not just the 936 no-titles paragraphs. Filter by the no-titles verdict ids
  from `data/labeled/all_combined.csv` before evaluating.
- Recomputing no-titles accuracy from that file yields figures a few points off the
  ones in `all_statistics_summary.csv`, so the two artifacts appear to have been
  generated from different runs. The summary file is the one cited above.
- GPT results are non-deterministic; re-running the notebooks will shift the numbers
  slightly.
- **`data/paragraphs/` covers a corpus whose sources are absent.** It holds extracted
  paragraphs for 4,560 verdicts (2,936 drugs + 1,624 weapon), but only 30 source DOCX
  files per corpus are committed — and just 25 of each overlap. The remaining ~4,510
  came from a local path on the original authors' machine that was never published.
  These CSVs are therefore the only surviving trace of that corpus and cannot be
  regenerated from this repository; `05_header_detection.ipynb` consumes them directly.
- The fine-tuned model weights are likewise absent (see Step 3), so the reported BERT
  metrics cannot be re-verified without retraining.

---

## Data

All verdict documents are real Israeli court judgments in Hebrew. Ground-truth labels
were assigned manually.
