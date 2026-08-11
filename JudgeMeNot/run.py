#!/usr/bin/env python
"""JudgeMeNot - unified CLI for training, evaluation, and analysis.

Usage:
    python run.py <command> [options]

Commands:
    train           Train LoRA adapters (CLM / Instruction / CLORA)
    eval            Evaluate adapters on test sets
    metrics         Add metrics (BLEU, ROUGE, BERTScore, ...) to eval CSVs
    rag-index       Build per-judge FAISS RAG indices
    merge           Merge LoRA adapters into base model weights
    ablation        Run data-size or LoRA-rank ablation study
    baselines       Run Wilcoxon baseline comparisons
    bootstrap       Run bootstrap significance analysis
    hparam          Hyperparameter grid search
    clf-data        Prepare per-judge classifier datasets
    clf-train       Train per-judge DictaBERT classifiers
    clf-eval        Run fooling evaluation (real vs generated)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ======================================================================
#  Subcommand: train
# ======================================================================

def cmd_train(args):
    from src.config import load_config
    from src.training.trainer import train

    cfg = load_config(args.config)
    train(cfg)


# ======================================================================
#  Subcommand: eval
# ======================================================================

def cmd_eval(args):
    from src.config import load_config
    from src.inference.generator import evaluate_adapters, evaluate_base_model

    cfg = load_config(args.config) if args.config else None

    rag_manager = None
    if args.rag_cache:
        from src.data.rag_store import RAGManager
        rag_manager = RAGManager(
            args.rag_cache,
            embedding_model=cfg.rag.embedding_model if cfg else "dicta-il/dictabert",
        )

    split_n, split_which = None, None
    if args.third:
        split_n, split_which = 3, args.third
    elif args.fifth:
        split_n, split_which = 5, args.fifth

    if args.base_only:
        evaluate_base_model(
            model_name=args.base_model or (cfg.model.name if cfg else "google/gemma-3-4b-it"),
            data_root=args.data_root,
            out_root=args.out_root or "outputs/eval",
            quantize=cfg.model.quantize if cfg else "nf4",
            batch_size=args.batch_size or (cfg.eval.batch_size if cfg else 4),
            compute_ppl=not args.no_ppl,
        )
    else:
        evaluate_adapters(
            models_root=args.models_root,
            data_root=args.data_root,
            out_root=args.out_root or "outputs/eval",
            base_model=args.base_model or (cfg.model.name if cfg else "google/gemma-3-4b-it"),
            method=args.method or (cfg.method if cfg else "instruction"),
            quantize=cfg.model.quantize if cfg else "nf4",
            batch_size=args.batch_size or (cfg.eval.batch_size if cfg else 4),
            cap_min=cfg.eval.cap_min if cfg else 1,
            cap_max=cfg.eval.cap_max if cfg else 1024,
            do_sample=args.do_sample,
            temperature=args.temperature,
            compute_ppl=not args.no_ppl,
            split_n=split_n,
            split_which=split_which,
            rag_manager=rag_manager,
            rag_k=cfg.rag.k if cfg else 3,
            rag_exact_threshold=cfg.rag.exact_match_threshold if cfg else 0.95,
            rag_context_modes=cfg.rag.context_modes if cfg and rag_manager else None,
        )


# ======================================================================
#  Subcommand: metrics
# ======================================================================

def cmd_metrics(args):
    metric_list = [m.strip() for m in args.metrics.split(",")]

    text_metrics = [m for m in metric_list if m in ("bleu", "rouge")]
    if text_metrics:
        from src.metrics.text_metrics import add_metrics_to_directory
        add_metrics_to_directory(
            args.input_dir, metrics=text_metrics,
            recursive=args.recursive, workers=args.workers,
        )

    if "bertscore" in metric_list:
        from src.metrics.bertscore import add_bertscore_to_directory
        add_bertscore_to_directory(
            args.input_dir,
            model_type=args.bertscore_model or "dicta-il/dictabert",
            ft_model_path=args.ft_model,
        )

    if "style" in metric_list:
        from src.metrics.style import add_style_metrics_to_csv
        from pathlib import Path
        for f in Path(args.input_dir).glob("*.csv"):
            add_style_metrics_to_csv(str(f))


# ======================================================================
#  Subcommand: rag-index
# ======================================================================

def cmd_rag_index(args):
    from src.data.rag_store import build_rag_index

    build_rag_index(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        embedding_model=args.embedding_model,
        data_source=args.data_source,
        force_rebuild=args.force,
        batch_size=args.batch_size,
    )


# ======================================================================
#  Subcommand: merge
# ======================================================================

def cmd_merge(args):
    from src.models.merge import batch_merge_adapters

    batch_merge_adapters(
        base_model_id=args.base_model,
        adapters_root=args.adapters_root,
        output_root=args.output,
        skip_existing=not args.force,
    )


# ======================================================================
#  Subcommand: ablation
# ======================================================================

def cmd_ablation(args):
    from src.config import load_config
    from src.analysis.ablation import run_ablation

    cfg = load_config(args.config)
    run_ablation(
        cfg=cfg,
        judge_name=args.judge,
        data_root=args.data_root or cfg.data.root,
        output_root=args.out_root or cfg.output.root,
        ablation_type=args.type,
        fractions=[float(x) for x in args.fractions.split(",")] if args.fractions else None,
        ranks=[int(x) for x in args.ranks.split(",")] if args.ranks else None,
    )


# ======================================================================
#  Subcommand: baselines
# ======================================================================

def cmd_baselines(args):
    from src.analysis.baselines import run_baseline_analysis

    run_baseline_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reference_model=args.reference,
        lower_is_better=args.lower_is_better,
    )


# ======================================================================
#  Subcommand: bootstrap
# ======================================================================

def cmd_bootstrap(args):
    from src.metrics.bootstrap import compute_bootstrap_summary, compute_delta_analysis

    if args.delta:
        compute_delta_analysis(
            eval_dir=args.input_dir,
            output_path=os.path.join(args.output_dir, "delta_analysis.csv"),
            n_boot=args.n_boot,
            ci=args.ci,
        )
    else:
        compute_bootstrap_summary(
            input_dir=args.input_dir,
            output_path=os.path.join(args.output_dir, "bootstrap_summary.csv"),
            n_boot=args.n_boot,
            ci=args.ci,
        )


# ======================================================================
#  Subcommand: hparam
# ======================================================================

def cmd_hparam(args):
    from src.config import load_config
    from src.training.trainer import hparam_search

    cfg = load_config(args.config)
    hparam_search(
        cfg=cfg,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        lora_ranks=[int(x) for x in args.ranks.split(",")] if args.ranks else None,
        learning_rates=[float(x) for x in args.lrs.split(",")] if args.lrs else None,
        output_root=args.out_root or "outputs/hparam_search",
    )


# ======================================================================
#  Subcommand: clf-data
# ======================================================================

def cmd_clf_data(args):
    from src.classifier.data import prepare_all_judges

    prepare_all_judges(
        data_root=args.data_root,
        chunks_root=args.chunks_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


# ======================================================================
#  Subcommand: clf-train
# ======================================================================

def cmd_clf_train(args):
    from src.classifier.train import train_all_classifiers

    train_all_classifiers(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model or "dicta-il/dictabert",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        patience=args.patience,
    )


# ======================================================================
#  Subcommand: clf-eval
# ======================================================================

def cmd_clf_eval(args):
    from src.classifier.evaluate import run_fooling_evaluation, compute_fooling_significance

    results = run_fooling_evaluation(
        classifier_dir=args.classifier_dir,
        eval_dir=args.eval_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        method_name=args.method_name,
        self_only=not args.all_cross,
        max_samples=args.max_samples,
        include_random_baseline=not args.no_baseline,
        include_ground_truth=not args.no_ground_truth,
        batch_size=args.batch_size,
    )

    if args.stats:
        compute_fooling_significance(
            results_path=os.path.join(args.output_dir, "results.csv"),
            output_path=os.path.join(args.output_dir, "statistical_tests.csv"),
        )


# ======================================================================
#  Argument parsing
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JudgeMeNot - personalized Hebrew legal Q&A generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    p = sub.add_parser("train", help="Train LoRA adapters")
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.set_defaults(func=cmd_train)

    # --- eval ---
    p = sub.add_parser("eval", help="Evaluate adapters on test sets")
    p.add_argument("--config", default=None, help="Experiment YAML config")
    p.add_argument("--models_root", help="Directory with adapter subdirectories")
    p.add_argument("--data_root", required=True, help="Directory with judge test data")
    p.add_argument("--out_root", default=None)
    p.add_argument("--base_model", default=None)
    p.add_argument("--method", default=None, choices=["instruction", "clm", "clora"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no_ppl", action="store_true", help="Skip perplexity computation")
    p.add_argument("--third", type=int, choices=[1, 2, 3])
    p.add_argument("--fifth", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--base_only", action="store_true", help="Eval base model without adapters")
    p.add_argument("--rag_cache", default=None, help="RAG cache directory for RAG eval")
    p.set_defaults(func=cmd_eval)

    # --- metrics ---
    p = sub.add_parser("metrics", help="Add metrics to evaluation CSVs")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--metrics", default="bleu,rouge,bertscore",
                   help="Comma-separated: bleu,rouge,bertscore,style")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--bertscore_model", default=None)
    p.add_argument("--ft_model", default=None, help="Fine-tuned BERT checkpoint")
    p.set_defaults(func=cmd_metrics)

    # --- rag-index ---
    p = sub.add_parser("rag-index", help="Build FAISS RAG indices")
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_dir", default="outputs/rag_cache")
    p.add_argument("--embedding_model", default="dicta-il/dictabert")
    p.add_argument("--data_source", choices=["train", "eval", "both"], default="both")
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.set_defaults(func=cmd_rag_index)

    # --- merge ---
    p = sub.add_parser("merge", help="Merge LoRA adapters into base model")
    p.add_argument("--adapters_root", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_merge)

    # --- ablation ---
    p = sub.add_parser("ablation", help="Run ablation study")
    p.add_argument("--config", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--type", choices=["data_size", "lora_rank"], default="data_size")
    p.add_argument("--data_root", default=None)
    p.add_argument("--out_root", default=None)
    p.add_argument("--fractions", default=None, help="Comma-separated fractions for data_size")
    p.add_argument("--ranks", default=None, help="Comma-separated LoRA ranks")
    p.set_defaults(func=cmd_ablation)

    # --- baselines ---
    p = sub.add_parser("baselines", help="Wilcoxon baseline comparisons")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--reference", default="clm-4b", help="Reference model name")
    p.add_argument("--lower_is_better", action="store_true")
    p.set_defaults(func=cmd_baselines)

    # --- bootstrap ---
    p = sub.add_parser("bootstrap", help="Bootstrap significance analysis")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--delta", action="store_true", help="Run delta analysis instead of summary")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.set_defaults(func=cmd_bootstrap)

    # --- hparam ---
    p = sub.add_parser("hparam", help="Hyperparameter grid search")
    p.add_argument("--config", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--eval_csv", default=None)
    p.add_argument("--ranks", default=None, help="Comma-separated LoRA ranks")
    p.add_argument("--lrs", default=None, help="Comma-separated learning rates")
    p.add_argument("--out_root", default=None)
    p.set_defaults(func=cmd_hparam)

    # --- clf-data ---
    p = sub.add_parser("clf-data", help="Prepare per-judge classifier datasets")
    p.add_argument("--data_root", default=None, help="Local QA data root (uses chunks if omitted)")
    p.add_argument("--chunks_root", default=None, help="Local chunk CSV root (default: data/chunks)")
    p.add_argument("--output_dir", default="outputs/classifier_datasets")
    p.add_argument("--max_samples", type=int, default=5000)
    p.set_defaults(func=cmd_clf_data)

    # --- clf-train ---
    p = sub.add_parser("clf-train", help="Train per-judge DictaBERT classifiers")
    p.add_argument("--dataset_dir", required=True, help="Dir with per-judge JSON splits")
    p.add_argument("--output_dir", required=True, help="Dir to save trained models")
    p.add_argument("--model", default=None, help="Base model (default: dicta-il/dictabert)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--patience", type=int, default=3)
    p.set_defaults(func=cmd_clf_train)

    # --- clf-eval ---
    p = sub.add_parser("clf-eval", help="Fooling evaluation (real vs generated)")
    p.add_argument("--classifier_dir", required=True, help="Dir with trained classifiers")
    p.add_argument("--eval_dir", required=True, help="Dir with generation CSVs")
    p.add_argument("--dataset_dir", required=True, help="Dir with classifier JSON splits")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method_name", default="default", help="Label for this method")
    p.add_argument("--all_cross", action="store_true", help="Include cross-judge CSVs")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_baseline", action="store_true", help="Skip random baseline")
    p.add_argument("--no_ground_truth", action="store_true", help="Skip ground-truth baseline")
    p.add_argument("--stats", action="store_true", help="Run Wilcoxon/t-test on results")
    p.set_defaults(func=cmd_clf_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
