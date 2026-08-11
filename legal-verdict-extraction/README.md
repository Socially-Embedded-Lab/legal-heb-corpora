# Legal Verdict Data Extraction Pipeline

Extracts structured data from Israeli court verdict documents for sentencing
prediction research. Four extraction stages — indictment facts, legal citations,
sentencing range — are merged into a unified per-verdict dataset covering 8,077
verdicts across three domains.

---

## Project Structure

```
legal-verdict-extraction/
├── notebooks/
│   └── 00_docx_to_csv.ipynb              # DOC/DOCX → CSV conversion
├── src/
│   ├── 1_extract_indictment_facts.py     # Indictment facts (gpt-4-turbo-preview)
│   ├── 2_extract_citations.py            # Citations (regex + GPT + heBERT)
│   ├── 3_extract_sentencing_range.py     # Sentencing range (gpt-4o, few-shot)
│   └── 4_create_unified_file.py          # Merge into unified outputs
├── models/
│   └── citation_classifier/              # Fine-tuned heBERT citation relevance model
│       ├── train_classifier.py           #   Training script
│       ├── training_data.csv             #   581 labeled paragraphs
│       ├── best_threshold.txt            #   Decision threshold + metrics
│       ├── config.json                   #   Model architecture
│       ├── tokenizer.json  vocab.txt     #   Hebrew tokenizer (30k vocab)
│       └── (model.safetensors)           #   NOT included — see "Model weights"
├── results/
│   ├── all_domains_unified.csv           # 8,077 verdicts (Git LFS)
│   └── all_domains_complete_only.csv     # 4,760 complete verdicts (Git LFS)
└── docs/
    ├── project_overview_he.docx          # Hebrew project write-up
    └── files_detailed.xlsx               # File inventory
```

---

## Outputs

| File | Rows | Description |
|------|-----:|-------------|
| `results/all_domains_unified.csv` | 8,077 | All verdicts, including incomplete ones |
| `results/all_domains_complete_only.csv` | 4,760 | Only verdicts with all four components |

`complete_only` is a strict subset of `unified` (verified: all 4,760 verdicts appear
in both). The per-domain intermediates that `4_create_unified_file.py` also emits —
`drugs_unified.csv`, `weapon_unified.csv`, `5k_unified.csv` — are **not** committed
here; regenerate them by running step 4.

### Data Schema

Both files share the same 12 columns:

| Column | Description |
|--------|-------------|
| `verdict` | Case identifier (e.g. `ת"פ 1234-05-21`) |
| `domain` | `drugs`, `weapon`, or `5k` |
| `indictment_facts` | Extracted indictment facts (GPT-cleaned) |
| `indictment_facts_raw` | Raw extracted facts (heuristic) |
| `citations_json` | JSON array of cited cases for sentencing policy |
| `citations_count` | Number of citations |
| `sentencing_classification` | `POSITIVE` / `NEGATIVE` (has a sentencing range) |
| `sentencing_sentence` | The sentence declaring the punishment range |
| `sentencing_confidence` | Extraction confidence level |
| `sentencing_range_low` | Lower bound (months) |
| `sentencing_range_high` | Upper bound (months) |
| `sentencing_range_str` | Human-readable range string |

---

## Data Coverage

Across all 8,077 verdicts:

| Component | Available | Coverage |
|-----------|----------:|---------:|
| `verdict` | 8,077 | 100% |
| `indictment_facts` | 7,997 | 99% |
| citations (`citations_count` > 0) | 5,104 | 63% |
| sentencing range (`sentencing_range_str`) | 6,265 | 78% |
| sentencing range (`sentencing_range_low`, numeric) | 5,224 | 65% |
| **All four components** | **4,760** | **59%** |

> **The two sentencing-range rows measure different things.** 6,265 verdicts have some
> range *string*, but only 5,224 have a parsed numeric lower bound — a 1,041-verdict
> gap where the range was detected but not successfully converted to months. For
> sentencing prediction the numeric bound is what matters, so treat 65% as the usable
> figure. The "all four components" count uses the string form.

### By domain

| Domain | Total | Complete | Percentage |
|--------|------:|---------:|-----------:|
| Drugs | 2,947 | 2,016 | 68% |
| Weapon | 1,640 | 1,247 | 76% |
| 5k (mixed) | 3,490 | 1,497 | 43% |

The 5k corpus is markedly less complete than the two curated domains, and it is also
the largest single contributor to the dataset — worth accounting for before pooling
all three.

---

## Pipeline

### Step 0 — Document Conversion

**`notebooks/00_docx_to_csv.ipynb`** — converts DOC/DOCX to structured CSV, using
LibreOffice (`unoconv`) for DOC→DOCX, then extracting paragraphs with metadata
(part name, text content).

Source verdict documents are **not** included in this repository. Set `DATA_ROOT`
to wherever yours live:

```bash
export DATA_ROOT=/path/to/verdict/corpus
```

**Output:** `verdict_csv/*.csv`

### Step 1 — Indictment Facts

**`src/1_extract_indictment_facts.py`** — identifies start/end sections with heuristic
patterns, extracts the relevant text, then sends it to **`gpt-4-turbo-preview`** for
clean extraction of conviction facts.

*Start patterns:* indictment, verdict, defendant convicted, facts.
*End patterns:* arguments, report, discussion.

### Step 2 — Citation Extraction

**`src/2_extract_citations.py`** — a three-stage hybrid:

1. Find legal citations by regex (`ע"פ`, `רע"פ`, `ת"פ`, …), e.g.
   `ע"פ 1234/21 פלוני נ' מדינת ישראל`
2. Filter to relevant sections and extract context with **`gpt-4-turbo-preview`**
3. Classify relevance with the fine-tuned heBERT model in `models/citation_classifier/`

### Step 3 — Sentencing Range

**`src/3_extract_sentencing_range.py`** — filters candidate sentences by regex and
section name, then uses **`gpt-4o`** few-shot to find the sentence where the judge
declares the range, and converts it to months.

Input paths are external to this repo; supply them via environment variables
(colon-separated):

```bash
export CSV_DIRS="/path/to/5k/verdict_csv:/path/to/drugs/verdict_csv"
export GT_FILES="/path/to/features_gt_weapon.csv"
```

### Step 4 — Unified File

**`src/4_create_unified_file.py`** — merges everything into `results/`.

---

## Citation Classifier

A **heBERT** (`avichr/heBERT`) model fine-tuned to judge whether a citation is
relevant to sentencing policy.

| Metric | Value |
|--------|------:|
| Decision threshold | 0.65 |
| PR-AUC | 0.9127 |
| F1 | 0.9032 |
| Precision | 0.8615 |
| Recall | 0.9492 |
| False positive rate | 0.0763 |

Architecture is `BertForSequenceClassification` — 12 layers, 768 hidden, 12 heads —
trained on the 581 labeled paragraphs in `training_data.csv`.

### Model weights

`model.safetensors` (438 MB) is **not committed here**, to keep this collection within
GitHub's Git LFS quota. Everything needed to reproduce it is present:

```bash
python models/citation_classifier/train_classifier.py \
  --model_name avichr/heBERT \
  --tokenizer_dir avichr/heBERT \
  --output_dir models/citation_classifier
```

Defaults: learning rate 5e-5, 2 epochs, batch size 8, max length 256. The original
weights remain available from the upstream repository via `git lfs pull`.

---

## Evaluation & Annotation

| Task | Evaluation method | Sample size |
|------|-------------------|-------------|
| Indictment facts | LLM as judge | 100 per domain |
| Citations | Manual annotation | ~200 citations |
| Sentencing range | Validation against ground truth | As available |

---

## Requirements

```
pandas>=1.5.0
openai>=1.0.0
tqdm
python-docx
transformers
torch
```

Set your API key in the environment — never in the source:

```bash
export OPENAI_API_KEY="your-key"
```

---

## Running the Pipeline

```bash
export OPENAI_API_KEY="your-key"
export DATA_ROOT=/path/to/verdict/corpus

jupyter notebook notebooks/00_docx_to_csv.ipynb   # Step 0
python src/1_extract_indictment_facts.py          # Step 1
python src/2_extract_citations.py                 # Step 2

export CSV_DIRS="/path/to/verdict_csv"
export GT_FILES="/path/to/ground_truth.csv"
python src/3_extract_sentencing_range.py          # Step 3

python src/4_create_unified_file.py               # Step 4
```

---

## Notes

- All verdict content is in Hebrew; documentation and column names are in English.
- The pipeline supports incremental processing with checkpoints.
- Model outputs are non-deterministic; re-running extraction shifts results slightly.
- Notebook execution outputs are preserved and still reference the original authors'
  absolute paths. That is a historical record of their run, not a live path.
