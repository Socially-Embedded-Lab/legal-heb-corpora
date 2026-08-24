# Innovation Authority Artifacts

A collection of research and engineering projects, vendored into a single repository.

Each project lives in its own top-level directory and keeps its original `README.md`,
structure, and history-free working tree.

---

## Projects

| Project | Description |
|---------|-------------|
| [classification-judgment](classification-judgment/) | Hebrew legal text classifier — classifies paragraphs from Israeli court verdicts into six semantic categories using fine-tuned Hebrew BERT models (AlephBERT, DictaBERT) and `gpt-4.1-mini` few-shot strategies. 4,269 labeled paragraphs; ~92% accuracy. |
| [multilabel-defendant-classification](multilabel-defendant-classification/) | Hebrew legal defendant attribution — multi-label classification of which defendant(s) a sentence in a criminal verdict refers to, using `gpt-4.1-mini` with structured Hebrew prompting. 5,159 annotated rows; 77-84% micro-F1. |
| [legal-verdict-extraction](legal-verdict-extraction/) | Structured data extraction from Israeli verdicts for sentencing prediction — indictment facts, legal citations and sentencing ranges across 8,077 verdicts, combining GPT extraction with a fine-tuned heBERT citation classifier (F1 0.9032). |
| [JudgeMeNot](JudgeMeNot/) | Replication code for the ACL 2026 paper on personalizing LLMs to emulate judicial reasoning in Hebrew — per-judge LoRA adapters over Gemma/DictaLM (CLM, instruction tuning, CLORA, RAG), evaluated by text metrics and per-judge DictaBERT discernment classifiers. **Incomplete: `src/data/` is missing.** |
| [speaker-identification](speaker-identification/) | Speaker attribution in Hebrew verdicts — classifies each sentence as spoken by the judge, defense, prosecution or probation service using `gpt-4.1` with paragraph context. 906-sentence gold set at 87.6% macro-F1; inter-annotator κ 0.86. |
| [legalAI](legalAI/) | **Previous-stage project** — criminal sentence classification for Israeli weapon- and drug-related verdicts via few-shot learning (SetFit and GPT). Uses a manually tagged dataset built with Ministry of Justice criminal-law experts, plus a ChatGPT-generated/refined dataset. Predates and is unrelated to the other projects above. |

---

## Layout

```
Innovation_Authority_Artifacts/
├── README.md                              # This file — the project index
├── .gitattributes                         # Git LFS tracking for large data files
├── classification-judgment/               # Vendored project (see its own README)
├── multilabel-defendant-classification/   # Vendored project (see its own README)
├── legal-verdict-extraction/              # Vendored project (see its own README)
├── JudgeMeNot/                            # Vendored project (see its own README)
├── speaker-identification/                # Locally delivered project
└── legalAI/                               # Vendored, previous-stage project (see its own README)
```

## Git LFS

`legal-verdict-extraction/results/*.csv` (181MB combined) is stored via Git LFS, since
GitHub hard-rejects files over 100MB. Cloning therefore needs LFS installed:

```bash
brew install git-lfs && git lfs install
git clone https://github.com/Maximbrg/Innovation_Authority_Artifacts
```

Without it you get 130-byte pointer files instead of the data. GitHub's free tier
allows 1GB of LFS storage and 1GB/month of bandwidth; keep an eye on this before
adding more large binaries.

---

## Adding a Project

Projects are vendored, not submoduled — the files are committed directly into this
repository so it stays self-contained and clonable in one step.

```bash
git clone https://github.com/<owner>/<repo>.git
rm -rf <repo>/.git
```

Then add a row to the [Projects](#projects) table above.

Projects delivered outside GitHub (archives, email attachments) are copied in the same
way.

### Local modifications

Vendored projects may be restructured after import — renamed directories, fixed
paths, corrected documentation. Each project's own README describes its current
layout; that layout is the source of truth, not the upstream repository's.

The notebook-and-data projects follow a shared layout — `config/`, `data/labeled/`,
`notebooks/`, `results/` — so they read consistently despite differing upstream
structures. `JudgeMeNot` is deliberately left alone: it is already an idiomatic Python
package with a CLI entry point and a config system, and reshaping it would break
imports for no gain.

- `classification-judgment` — reorganized into a `data/` hierarchy with numbered
  notebooks; machine-specific absolute paths rewritten to be repo-relative.
- `multilabel-defendant-classification` — restructured to the shared layout, typo'd
  directory names corrected, committed `.DS_Store` files removed, and three bugs fixed
  that prevented the notebook from running from a fresh kernel.
- `legal-verdict-extraction` — restructured to the shared layout; credentials moved out
  of the source and into environment variables, along with the authors' absolute paths;
  the 438MB `model.safetensors` omitted (regenerable from the committed training script
  and data).
- `JudgeMeNot` — structure preserved. Fixed a `merged_root` typo in `configs/paths.yaml`
  that pointed CLORA at a directory the merge step never writes, and documented the
  missing `src/data/` package and exactly which commands it breaks.
- `speaker-identification` — delivered as a Hebrew-named archive with no README. Names
  translated to English, a doubled `.ipynb.ipynb` extension fixed, and a README written
  from scratch. The private email that carried the archive was **excluded**; its project
  write-up attachment was kept in `docs/`. Two 0-byte notebooks were omitted and
  documented rather than shipped as empty files.
- `legalAI` — added as-is, no restructuring. A note was prepended to its README
  flagging it as a previous-stage project from
  [Maximbrg/legalAI](https://github.com/Maximbrg/legalAI) that predates and is
  unrelated to the rest of this repository.

### Re-syncing with upstream

Because the nested `.git` is removed, a vendored project has no live link to its
origin. To pull in upstream changes, re-clone into a temporary directory and copy
the updated files over — then reapply any local restructuring:

```bash
git clone https://github.com/<owner>/<repo>.git /tmp/<repo>-fresh
rsync -a --delete --exclude '.git' /tmp/<repo>-fresh/ <repo>/
```

Review the resulting diff before committing.
