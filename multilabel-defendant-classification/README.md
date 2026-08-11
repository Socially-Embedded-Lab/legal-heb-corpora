# Hebrew Legal Defendant Attribution (Multi-Label)

Predicts **which defendant(s)** a given sentence in a Hebrew criminal verdict refers
to, using `gpt-4.1-mini` with structured Hebrew prompting. It addresses the complexity
of legal Hebrew, where subjects are often implied or referred to by generic titles.

---

## Task Overview

Multi-label classification performed at the row (sentence) level.

* **Input:** a single sentence ("snippet"), case metadata (`case_file_name`,
  `num_defendants`), and the full case text for context mapping.
* **Output:** a compact digit string representing the set of defendants referenced
  (e.g. `12` for Defendants 1 and 2).

---

## The Challenge: Hebrew Legal Reality

Hebrew verdicts present unique NLP challenges around coreference resolution:

* **Implicit references:** `"הנאשם"` (The Defendant) used without a number. Identity
  depends on definitions made earlier in the document (e.g. `"נאשם מס' 1 - פלוני"`).
* **Plural ambiguity:** `"הנאשמים"` (The Defendants) usually means all of them, but can
  refer to a specific subset based on the paragraph's theme (e.g. a particular count in
  the indictment).
* **Contextual inference:** sentences about probation reports or personal circumstances
  often name nobody, yet clearly refer to one individual given preceding sentences.

### Solution Strategy

1. **Strict output formatting** — force the model into a consistent digit-based schema.
2. **Structured Hebrew prompting** — instruct the model in the domain language.
3. **Full-case mapping** — supply the entire verdict text so the model can resolve
   "Defendant 1 = [Name]" mappings.

---

## Project Structure

```
multilabel-defendant-classification/
├── config/
│   └── prompts.yaml                          # System + user prompt templates
├── data/
│   └── labeled/                              # Manually annotated gold standard
│       ├── annotation_drug_defendants.csv    #   3,246 rows
│       └── annotation_weapon_defendants.csv  #   1,913 rows
├── notebooks/
│   └── 01_predict_defendants.ipynb           # Inference + evaluation
└── results/
    ├── predictions/                          # Per-row model predictions
    │   ├── defendants_pred_drugs.csv         #   3,246 rows
    │   └── defendants_pred_weapon.csv        #   1,913 rows
    └── metrics/                              # Aggregated scores
        ├── eval_drugs.csv
        └── eval_weapon.csv
```

---

## Dataset Format

### Gold standard (`data/labeled/`)

| Column | Description |
| --- | --- |
| `case_file_name` | Unique case identifier |
| `text` | The specific snippet/sentence to classify |
| `defendants_str` | **Gold label:** the ground-truth defendant set |
| `num_defendants` | Total count of defendants in that case |

### Predictions (`results/predictions/`)

The same four columns plus `pred`, the model's predicted defendant set. `pred` is
produced during inference — it is not part of the input schema.

### Label Encoding

The defendant set is a sequence of digits:

* `0` — none
* `1` — Defendant 1
* `23` — Defendants 2 and 3
* `123` — all defendants (in a 3-defendant case)

---

## Setup

```bash
pip install pandas pyyaml openai
export OPENAI_API_KEY="your_key_here"
```

Requires Python 3.10+.

Paths default to the repo layout, so the notebook runs as-is from `notebooks/`.
Override any of them via environment variables to point elsewhere:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — (required) | OpenAI credentials |
| `INPUT_CSV` | `../data/labeled/annotation_drug_defendants.csv` | Rows to classify |
| `OUTPUT_DIR` | `../results` | Where predictions, metrics and logs are written |
| `PROMPTS_YAML` | `../config/prompts.yaml` | Prompt template file |

Switch corpora by pointing `INPUT_CSV` at the weapon annotations.

---

## Pipeline

Run [`notebooks/01_predict_defendants.ipynb`](notebooks/01_predict_defendants.ipynb).

### 1. Build full-case text map

Concatenate all rows sharing a `case_file_name` into a single string, so the model can
see the head of the verdict where defendants are formally introduced. Case text is
truncated at 50,000 characters, individual snippets at 12,000.

### 2. Inference

Each row is classified by **`gpt-4.1-mini`**, injecting the snippet, its context and
the full-case map into the YAML prompt template. Calls are spaced by 1.2s with up to
5 retries and jittered backoff to stay within rate limits.

### 3. Evaluation

Because the task is multi-label, four complementary metrics are reported:

* **Micro precision / recall / F1** — aggregated across all individual defendant
  decisions.
* **Exact match accuracy** — the entire predicted set must be correct.
* **Average Jaccard** — intersection over union, giving partial credit. If the truth is
  `12` and the prediction is `1`, Jaccard scores 0.5 where exact match scores 0.

---

## Results

Scores are broken down by `N`, the number of defendants in the case.

### Drugs corpus (3,246 rows)

| Subset | Micro-P | Micro-R | Micro-F1 | Exact match | Avg Jaccard |
|--------|--------:|--------:|---------:|------------:|------------:|
| Overall | 74.99% | 78.75% | 76.82% | 71.90% | 73.99% |
| N=2 | 81.26% | 82.68% | 81.97% | 76.17% | 78.12% |
| N=3 | 71.48% | 76.10% | 73.72% | 69.01% | 71.03% |

### Weapon corpus (1,913 rows)

| Subset | Micro-P | Micro-R | Micro-F1 | Exact match | Avg Jaccard |
|--------|--------:|--------:|---------:|------------:|------------:|
| Overall | 84.30% | 83.33% | 83.81% | 77.11% | 79.60% |
| N=2 | 82.40% | 82.62% | 82.51% | 75.73% | 77.56% |
| N=3 | 86.89% | 84.28% | 85.56% | 79.44% | 83.03% |

Two patterns stand out. The weapon corpus scores ~7 points higher on micro-F1 than
drugs despite being smaller. And the corpora behave in opposite directions as defendant
count grows: on drugs, 3-defendant cases are ~8 points *worse* than 2-defendant ones —
the expected effect, since more defendants means more ways to be wrong — while on
weapons they are ~3 points *better*. That inversion is unexplained and worth
investigating before treating the weapon numbers as a ceiling.

Exact match trails micro-F1 by 5-7 points throughout, which is what partial-credit
metrics are for: many errors are near-misses on one defendant rather than wholly wrong
sets.

---

## Notes

* Model outputs are non-deterministic; re-running inference will shift the numbers
  slightly.
* `results/predictions/` is named per corpus, but the notebook writes timestamped files
  (`defendants_pred_<timestamp>.csv`); the committed copies were renamed by hand.
