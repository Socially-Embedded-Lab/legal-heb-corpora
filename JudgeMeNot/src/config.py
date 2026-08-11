"""Configuration dataclasses and YAML loading with environment-aware path resolution."""

from __future__ import annotations 

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str = "google/gemma-3-4b-it"
    quantize: str = "nf4"                   # nf4 | awq | none
    seq_length: int = 512
    attn_implementation: str = "eager"

@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"      # "all-linear" or list

@dataclass
class TrainingConfig:
    method: str = "instruction"             # clm | instruction | clora
    epochs: int = 15
    batch_size: int = 2
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    scheduler: str = "constant"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    save_steps: int = 100
    logging_steps: int = 20
    patience: int = 2
    packing: bool = False
    seed: int = 42
    optim: str = "adamw_torch_fused"
    max_steps: int = -1                     # -1 = use epochs

@dataclass
class DataConfig:
    root: str = ""
    datasets: Optional[List[str]] = None    # Judge names (CLM local) or HF IDs, or None (scan dirs)
    chunks_root: Optional[str] = None       # local CSV root for CLM chunk data
    merged_root: Optional[str] = None       # pre-merged base dir (CLORA)
    data_source: str = "both"               # train | eval | both (for RAG indexing)

@dataclass
class EvalConfig:
    batch_size: int = 4
    do_sample: bool = False
    temperature: float = 0.0
    cap_min: int = 1
    cap_max: int = 1024
    compute_ppl: bool = True
    split: Optional[int] = None             # third (1-3) or fifth (1-5)
    split_mode: str = "third"               # third | fifth
    prompt_pct: float = 0.1                 # CLM: fraction of text used as prompt

@dataclass
class RAGConfig:
    cache_dir: str = "outputs/rag_cache"
    embedding_model: str = "dicta-il/dictabert"
    k: int = 3
    exact_match_threshold: float = 0.95
    context_modes: List[str] = field(default_factory=lambda: ["own_context", "others_context"])

@dataclass
class AblationConfig:
    ablation_type: str = "data_size"        # data_size | lora_rank
    fractions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    ranks: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    judge: str = ""
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "bertscore"])

@dataclass
class OutputConfig:
    root: str = "outputs"
    tensorboard: str = "outputs/runs"

@dataclass
class ExperimentConfig:
    """Top-level config that composes all sub-configs."""
    method: str = "instruction"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
#  Path resolution
# ---------------------------------------------------------------------------

_PATHS_CACHE: Optional[Dict] = None


def _load_paths(paths_yaml: Optional[str] = None) -> Dict:
    """Load paths.yaml and return the dict for the active environment."""
    global _PATHS_CACHE
    if _PATHS_CACHE is not None:
        return _PATHS_CACHE

    if paths_yaml is None:
        candidates = [
            Path("configs/paths.yaml"),
            Path(__file__).resolve().parent.parent / "configs" / "paths.yaml",
        ]
        for c in candidates:
            if c.exists():
                paths_yaml = str(c)
                break

    if paths_yaml is None or not Path(paths_yaml).exists():
        _PATHS_CACHE = {}
        return _PATHS_CACHE

    with open(paths_yaml, "r") as f:
        all_paths = yaml.safe_load(f) or {}

    _PATHS_CACHE = all_paths.get("local", {})
    return _PATHS_CACHE


def _resolve_vars(value: str, paths: Dict) -> str:
    """Replace ${paths.key} placeholders with actual values."""
    def _replacer(m):
        key = m.group(1)
        parts = key.split(".")
        if parts[0] == "paths" and len(parts) == 2:
            return str(paths.get(parts[1], m.group(0)))
        return m.group(0)

    return re.sub(r"\$\{([^}]+)\}", _replacer, value)


def _resolve_dict(d: dict, paths: Dict) -> dict:
    """Recursively resolve ${paths.*} placeholders in a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = _resolve_vars(v, paths)
        elif isinstance(v, dict):
            out[k] = _resolve_dict(v, paths)
        elif isinstance(v, list):
            out[k] = [_resolve_vars(x, paths) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, d: dict):
    """Populate a dataclass from a dict, ignoring extra keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in field_names})


def load_config(
    yaml_path: str,
    overrides: Optional[Dict] = None,
) -> ExperimentConfig:
    """
    Load an experiment YAML config, resolve path variables, and return
    a typed ExperimentConfig.

    Parameters
    ----------
    yaml_path : str
        Path to the experiment YAML file.
    overrides : dict, optional
        Flat key=value overrides applied after YAML loading.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    paths = _load_paths()
    raw = _resolve_dict(raw, paths)

    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            target = raw
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = v

    method = raw.get("method", "instruction")

    cfg = ExperimentConfig(
        method=method,
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        lora=_dict_to_dataclass(LoraConfig, raw.get("lora", {})),
        training=_dict_to_dataclass(TrainingConfig, {**raw.get("training", {}), "method": method}),
        data=_dict_to_dataclass(DataConfig, raw.get("data", {})),
        eval=_dict_to_dataclass(EvalConfig, raw.get("eval", {})),
        rag=_dict_to_dataclass(RAGConfig, raw.get("rag", {})),
        ablation=_dict_to_dataclass(AblationConfig, raw.get("ablation", {})),
        output=_dict_to_dataclass(OutputConfig, raw.get("output", {})),
    )
    return cfg
