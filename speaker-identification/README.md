# Speaker Identification in Hebrew Legal Documents

Identifies **who is speaking** in a given sentence of an Israeli court verdict — the
judge, the defense, the prosecution, or the probation service. Each sentence is
classified by an LLM that reads the surrounding paragraph as context, since Hebrew
legal prose frequently reports one party's claims inside another's narration.

Originally delivered as a Hebrew-named archive (`זיהוי דוברים`); directory and file
names have been translated to English here. The original Hebrew project write-up is
preserved in [`docs/`](docs/).

---

## Task

Given a target sentence and the paragraph containing it, assign exactly one speaker
label from a closed set:

| Hebrew label | English | Gold count |
|--------------|---------|-----------:|
| `שופט` | Judge | 335 |
| `הגנה` | Defense | 307 |
| `תביעה` | Prosecution | 171 |
| `שירות המבחן` | Probation Service | 85 |
| `נאשם` | Defendant | 6 |

`נאשם` (Defendant) is not part of the intended scheme — the notebook's `LABEL_MAP`
folds it into `הגנה` (Defense). It survives in the raw gold file on 6 rows, and that
matters for the reported scores (see [Results](#results)).

The core difficulty is that a sentence's own wording often misleads: the judge
routinely quotes, paraphrases, or summarises the parties' arguments. The system prompt
therefore instructs the model to read the full paragraph first and to let paragraph
context override sentence-level phrasing.

---

## Project Structure

```
speaker-identification/
├── notebooks/
│   └── 01_speaker_identification_llm.ipynb   # Inference + evaluation pipeline
├── data/
│   ├── speaker_dataset_full.csv              # 30,062 sentences · 181 cases (unlabeled)
│   ├── speaker_dataset_clean.csv             # 20,945 sentences · 135 cases (unlabeled)
│   ├── labeled/
│   │   └── speaker_dataset_gold.csv          # 906 annotated sentences · 25 cases
│   └── annotations/                          # Inter-annotator agreement sample
│       ├── sample50_annotator_guy.xlsx       #   50 sentences, annotator A
│       └── sample50_annotator_mor.xlsx       #   50 sentences, annotator B
├── results/
│   ├── speaker_dataset_inference.csv         # 20,945 predictions over the clean set
│   └── agreement/
│       ├── agreement_summary.txt             # Percent agreement + Cohen's kappa
│       ├── confusion_matrix.csv              # Annotator A vs B
│       └── merged_labels.csv                 # Side-by-side annotator labels
└── docs/
    └── speaker_identification_legal_documents_he.docx   # Hebrew project write-up
```

### How the datasets relate

`full` (30,062 sentences, 181 cases) is filtered down to `clean` (20,945 sentences,
135 cases) — the corpus actually run through inference. Both carry an empty `speaker1`
column: they are annotation *inputs*, not labeled data.

`gold` (906 sentences, 25 cases) is the manually annotated evaluation set and the only
file with real labels. Note that only 439 of its 906 sentences appear in `clean`, so
the two were not drawn from the same snapshot — worth confirming before treating gold
as a clean subset of the inference corpus.

---

## Data Schema

| Column | Present in | Description |
|--------|-----------|-------------|
| `case_id` | all | Verdict identifier |
| `paragraph_text` | all | Full paragraph, given to the model as context |
| `sentence_text` | all | The target sentence to classify |
| `speaker1` | all | Gold label (populated only in `gold`) |
| `speaker2` | gold | Second-annotator column — **present but entirely empty** |
| `pred_label` | gold | Model prediction |
| `pred_confidence`, `votes` | gold | Confidence and vote counts |
| `pred_speaker` | inference | Model prediction over the clean corpus |

---

## Method

Classification runs on **`gpt-4.1`** at temperature 0.2, with a Hebrew system prompt
that:

1. instructs the model to read and understand the whole paragraph before classifying;
2. states explicitly that paragraph context overrides sentence-level phrasing when the
   two conflict;
3. constrains output to exactly one of the four canonical labels;
4. requires the answer as bare JSON.

A hard-examples block supports few-shot prompting for ambiguous cases. The API key is
read from the environment — the notebook hardcodes nothing.

---

## Results

### Speaker classification (906-sentence gold set)

Computed from `data/labeled/speaker_dataset_gold.csv`, which ships with predictions:

| | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Raw labels, as stored | 87.94% | **69.91%** | 87.67% |
| With `LABEL_MAP` applied | 88.50% | **87.57%** | 88.54% |

**The 18-point macro-F1 gap is an artifact, not a finding.** Six rows carry the
stray `נאשם` label; the model never predicts it, so that class scores F1 0.00 and drags
the unweighted mean down across a 5-class average. Applying the notebook's own
`LABEL_MAP` — which folds `נאשם` into `הגנה` — collapses the problem to the intended
4 classes and macro-F1 jumps to 87.57%. **Quote the normalized figures**; the raw ones
measure a labeling inconsistency rather than model quality.

Per-class, after normalization:

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| `הגנה` (Defense) | 0.95 | 0.93 | 0.94 | 313 |
| `תביעה` (Prosecution) | 0.88 | 0.88 | 0.88 | 171 |
| `שופט` (Judge) | 0.87 | 0.83 | 0.85 | 335 |
| `שירות המבחן` (Probation) | 0.75 | 0.93 | 0.83 | 85 |

Probation Service is the weakest class: recall 0.93 against precision 0.75, meaning the
model over-assigns it. Judge has the inverse profile — the most frequent class, yet the
lowest recall, consistent with the core difficulty that judge-narrated text often
reports other parties' arguments.

### Inter-annotator agreement (50-sentence sample)

| Metric | Value |
|--------|------:|
| Sentences compared | 50 |
| Percent agreement | 90.00% |
| Cohen's κ | **0.8575** |

κ above 0.8 is conventionally read as strong agreement, so the label scheme is
reproducible between annotators. Of the five disagreements, four involve `שופט`
(Judge) — the same confusion the model exhibits, suggesting the difficulty is inherent
to the task rather than a model weakness.

---

## Setup

```bash
pip install openai pandas numpy tqdm scikit-learn matplotlib
export OPENAI_API_KEY="your-key"
```

Then run [`notebooks/01_speaker_identification_llm.ipynb`](notebooks/01_speaker_identification_llm.ipynb).

The notebook expects two paths set in its own cells — `DATA_PATH` for the input CSV and
`PREDS_PATH` for saved predictions. Point them at `data/labeled/speaker_dataset_gold.csv`
and a predictions file respectively; the defaults refer to filenames not present here.

---

## Known Gaps

- **Two pipeline stages have no code.** The delivered archive contained
  `agreement.ipynb` and `creating_dataset.ipynb` as **0-byte files** — verified empty in
  the original zip as well, so this is not an extraction error. Dataset construction
  (`full` → `clean`) and the agreement computation are therefore undocumented in code,
  though their *outputs* are present in `data/` and `results/agreement/`. Request these
  from the authors.
- **`speaker2` is empty.** The gold file reserves a second-annotator column but no
  values were recorded; agreement was computed separately from the two spreadsheets in
  `data/annotations/`.
- **`gold` is not a subset of `clean`** (439 of 906 sentences overlap), so the
  evaluation set and inference corpus may come from different preprocessing runs.
- Model outputs are non-deterministic; re-running inference will shift results slightly.
