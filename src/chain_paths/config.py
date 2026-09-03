"""Configuration settings and runtime overrides for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Deterministic execution
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Dataclass-backed configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "pipeline.yaml"

# Shared dataset metadata ---------------------------------------------------

WINDOWS_APT_COMBINED_FILENAME = "combined.csv"


def _default_params() -> Dict[str, Any]:
    params = {
        'lambda': 5.0,
        'eta': 0.7,
        'kappa': 10.0,
        'tau': 5.0,
        'k_history': 3,
        'rho': 2.0,
        'RANDOM_SEED': RANDOM_SEED,
        'beta': 1.0,
        'tau_decay': 365,
        # Three-pillar model: LSTM + Tactic Prior + Stealth Score
        # Removed (redundant with LSTM): count, embedding, prior, causal, mcdm
        'w_count': 0.0,      # REMOVED - subsumed by LSTM
        'w_emb': 0.0,        # REMOVED - LSTM uses embeddings internally
        'w_prior': 0.0,      # REMOVED - subset of LSTM output
        'w_stealth': 0.1,    # Sigma-rule detection difficulty
        'w_tactic': 0.1,     # Kill-chain tactic transition prior
        'w_causal': 0.0,     # REMOVED - needs richer annotations
        'w_bilstm': 1.0,     # PRIMARY - neural sequence model
        'w_lstm': 1.0,       # Unidirectional LSTM (alternate neural model)
        'epsilon': 1e-9,
        'bootstrap_rounds': 1000,
        'significance_alpha': 0.05,
        'path_samples': 256,
        'use_tactic_filtering': True,
        'tactic_threshold': 0.01,
        'branching_factor': 3,
        'techniques_per_tactic': 5,
        'diversity_weight': 0.3,
        'branch_selection_mode': 'top_k',
        'use_platform_filtering': True,
        'allowed_platforms': ['windows', 'active directory'],
    }
    # Three-pillar log-linear component weights (LSTM + Tactic + Stealth)
    component_weights = {
        'bias': 0.0,
        'log_bilstm': 1.0,    # PRIMARY - neural sequence model
        'log_tactic': 1.0,    # Kill-chain transitions
        'log_stealth': 1.0,   # Detection difficulty
        # Legacy (set to 0, kept for backward compatibility)
        'log_interp': 0.0,
        'log_count': 0.0,
        'log_embedding': 0.0,
        'log_prior': 0.0,
    }
    params['component_weights'] = component_weights
    return params


def _word2vec_params() -> Dict[str, Any]:
    return {
        'vector_size': 128,
        'window': 4,
        'min_count': 5,
        'sg': 1,
        'negative': 10,
        'epochs': 12,
        'seed': RANDOM_SEED,
        'workers': os.cpu_count(),
    }


def _beam_search_params() -> Dict[str, Any]:
    return {
        'beam_width': 200,
        'max_depth': 6,
        'top_k_paths': 100,
        'top_n_candidates': 200,
    }


def _eval_params() -> Dict[str, Any]:
    return {
        'train_test_split': 0.8,
        'temporal_split': True,
        'k_values': [1, 3, 5, 10, 20],
    }

PATH_FIELDS: Tuple[str, ...] = (
    'data_dir',
    'outputs_dir',
    'notebooks_dir',
    'sightings_csv',
    'windows_apt_dataset_dir',
    'sigma_rules_dir',
    'sigma_stealth_csv',
    'sigma_rule_details_json',
    'sequences_parquet',
    'sequences_legacy_parquet',
    'sequences_windows_apt_parquet',
    'enterprise_attack_json',
    'counts_unigram_parquet',
    'counts_bigram_parquet',
    'counts_trigram_parquet',
    'tech_embeddings_npy',
    'tech_index_csv',
    'word2vec_model',
    'paths_top_k_json',
    'metrics_json',
    'results_notebook',
    'component_weights_json',
    'tactic_mapping_json',
    'tactic_transitions_json',
    'ontology_json',
    'ontology_rules_json',
    'causal_annotations_json',
)


@dataclass
class PipelineConfig:
    """Serializable configuration covering paths and experiment knobs."""

    repo_root: Path = REPO_ROOT
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs")
    notebooks_dir: Path = field(default_factory=lambda: REPO_ROOT / "notebooks")
    sightings_csv: Path = field(default_factory=lambda: REPO_ROOT / "datasets" / "sightings_v2_public.csv")
    windows_apt_dataset_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "datasets" / "Windows-APT 2025 A Dataset for APT-Inspired Attack"
    )
    sigma_rules_dir: Path = field(default_factory=lambda: REPO_ROOT / "sigma-rules" / "windows")
    sigma_stealth_csv: Path = field(default_factory=lambda: REPO_ROOT / "data" / "sigma_stealth.csv")
    sigma_rule_details_json: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "sigma_rule_details.json")
    sequences_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "sequences.parquet")
    sequences_legacy_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "sequences_legacy.parquet")
    sequences_windows_apt_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "sequences_windows_apt.parquet")
    enterprise_attack_json: Path = field(default_factory=lambda: REPO_ROOT / "data" / "enterprise-attack.json")
    counts_unigram_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "counts_unigram.parquet")
    counts_bigram_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "counts_bigram.parquet")
    counts_trigram_parquet: Path = field(default_factory=lambda: REPO_ROOT / "data" / "counts_trigram.parquet")
    tech_embeddings_npy: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "tech_embeddings.npy")
    tech_index_csv: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "tech_index.csv")
    word2vec_model: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "word2vec.model")
    paths_top_k_json: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "paths_topK.json")
    metrics_json: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "metrics.json")
    results_notebook: Path = field(default_factory=lambda: REPO_ROOT / "notebooks" / "results.ipynb")
    component_weights_json: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "component_weights.json")
    tactic_mapping_json: Path = field(default_factory=lambda: REPO_ROOT / "data" / "technique_tactic_map.json")
    tactic_transitions_json: Path = field(default_factory=lambda: REPO_ROOT / "data" / "tactic_transitions.json")
    ontology_json: Path = field(default_factory=lambda: REPO_ROOT / "data" / "ontology.json")
    ontology_rules_json: Path = field(default_factory=lambda: REPO_ROOT / "data" / "ontology_rules.json")
    causal_annotations_json: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "causal_annotations.json")
    default_params: Dict[str, Any] = field(default_factory=_default_params)
    word2vec_params: Dict[str, Any] = field(default_factory=_word2vec_params)
    beam_search_params: Dict[str, Any] = field(default_factory=_beam_search_params)
    eval_params: Dict[str, Any] = field(default_factory=_eval_params)
    model_types: Tuple[str, ...] = ('ngram', 'hmm', 'bilstm', 'lstm', 'gru', 'tcn', 'transformer')
    default_model_type: str = 'ngram'
    hmm_default_config: Dict[str, Any] = field(default_factory=lambda: {'n_states': 4, 'smoothing': 1.0})
    bilstm_default_config: Dict[str, Any] = field(default_factory=lambda: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.1})
    lstm_default_config: Dict[str, Any] = field(default_factory=lambda: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.1})
    gru_default_config: Dict[str, Any] = field(default_factory=lambda: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.1})
    tcn_default_config: Dict[str, Any] = field(default_factory=lambda: {'channels': 256, 'depth': 4, 'dropout': 0.1})
    transformer_default_config: Dict[str, Any] = field(default_factory=lambda: {'d_model': 256, 'num_layers': 4, 'num_heads': 4, 'dropout': 0.1})
    data_sources: Dict[str, Any] = field(
        default_factory=lambda: {
            'choices': ("legacy", "windows-apt", "both"),
            'default': "both",
        }
    )

    def apply_overrides(self, overrides: Optional[Mapping[str, Any]]) -> None:
        if not overrides:
            return

        paths = overrides.get('paths') if isinstance(overrides, Mapping) else None
        if isinstance(paths, Mapping):
            for key, value in paths.items():
                self._set_path_field(key, value)

        params = overrides.get('params') if isinstance(overrides, Mapping) else None
        if isinstance(params, Mapping):
            if 'default' in params and isinstance(params['default'], Mapping):
                self.default_params.update(params['default'])
            if 'word2vec' in params and isinstance(params['word2vec'], Mapping):
                self.word2vec_params.update(params['word2vec'])
            if 'beam_search' in params and isinstance(params['beam_search'], Mapping):
                self.beam_search_params.update(params['beam_search'])
            if 'eval' in params and isinstance(params['eval'], Mapping):
                self.eval_params.update(params['eval'])

        data_sources = overrides.get('data_sources') if isinstance(overrides, Mapping) else None
        if isinstance(data_sources, Mapping):
            self.data_sources.update(data_sources)

        for key, value in overrides.items():
            if key in {'paths', 'params', 'data_sources'}:
                continue
            if key in PATH_FIELDS:
                self._set_path_field(key, value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def _set_path_field(self, key: str, value: Union[str, Path]) -> None:
        if key not in PATH_FIELDS:
            return
        path = Path(value)
        if not path.is_absolute():
            path = self.repo_root / path
        setattr(self, key, path)

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]]) -> "PipelineConfig":
        config = cls()
        if path is None:
            return config

        path = Path(path)
        if not path.exists():
            return config

        with path.open('r', encoding='utf-8') as handle:
            overrides = yaml.safe_load(handle) or {}
        if not isinstance(overrides, Mapping):
            return config

        config.apply_overrides(overrides)
        return config


def _load_initial_config() -> PipelineConfig:
    env_path = os.environ.get('CHAIN_PATHS_CONFIG')
    if env_path:
        return PipelineConfig.from_yaml(env_path)
    if DEFAULT_CONFIG_PATH.exists():
        return PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)
    return PipelineConfig()


_PIPELINE_CONFIG: PipelineConfig = _load_initial_config()


def get_pipeline_config() -> PipelineConfig:
    """Return the active pipeline configuration."""

    return _PIPELINE_CONFIG


def configure_pipeline(config_path: Optional[Union[str, Path]] = None,
                      overrides: Optional[Mapping[str, Any]] = None) -> PipelineConfig:
    """Load configuration from disk and refresh exported constants."""

    global _PIPELINE_CONFIG
    if config_path:
        _PIPELINE_CONFIG = PipelineConfig.from_yaml(config_path)
    elif overrides:
        _PIPELINE_CONFIG = PipelineConfig()
    if overrides:
        _PIPELINE_CONFIG.apply_overrides(overrides)
    _refresh_exports()
    return _PIPELINE_CONFIG


def _refresh_exports() -> None:
    global DATA_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR
    global SIGHTINGS_CSV, WINDOWS_APT_DATASET_DIR
    global SIGMA_RULES_DIR, SIGMA_STEALTH_CSV, SIGMA_RULE_DETAILS_JSON
    global SEQUENCES_PARQUET, SEQUENCES_LEGACY_PARQUET, SEQUENCES_WINDOWS_APT_PARQUET
    global ENTERPRISE_ATTACK_JSON
    global COUNTS_UNIGRAM_PARQUET, COUNTS_BIGRAM_PARQUET, COUNTS_TRIGRAM_PARQUET
    global TECH_EMBEDDINGS_NPY, TECH_INDEX_CSV, WORD2VEC_MODEL
    global PATHS_TOP_K_JSON, METRICS_JSON
    global RESULTS_NOTEBOOK, DEFAULT_PARAMS, WORD2VEC_PARAMS
    global BEAM_SEARCH_PARAMS, EVAL_PARAMS, COMPONENT_WEIGHTS_JSON
    global MODEL_TYPES, DEFAULT_MODEL_TYPE
    global HMM_DEFAULT_CONFIG, BILSTM_DEFAULT_CONFIG, LSTM_DEFAULT_CONFIG, GRU_DEFAULT_CONFIG
    global TCN_DEFAULT_CONFIG, TRANSFORMER_DEFAULT_CONFIG
    global DATA_SOURCE_CHOICES, DEFAULT_DATA_SOURCE, SEQUENCE_OUTPUT_MAP
    global TACTIC_MAPPING_JSON, TACTIC_TRANSITIONS_JSON
    global ONTOLOGY_JSON, ONTOLOGY_RULES_JSON, CAUSAL_ANNOTATIONS_JSON

    cfg = _PIPELINE_CONFIG
    DATA_DIR = cfg.data_dir
    OUTPUTS_DIR = cfg.outputs_dir
    NOTEBOOKS_DIR = cfg.notebooks_dir
    SIGHTINGS_CSV = cfg.sightings_csv
    WINDOWS_APT_DATASET_DIR = cfg.windows_apt_dataset_dir
    SIGMA_RULES_DIR = cfg.sigma_rules_dir
    SIGMA_STEALTH_CSV = cfg.sigma_stealth_csv
    SIGMA_RULE_DETAILS_JSON = cfg.sigma_rule_details_json
    SEQUENCES_PARQUET = cfg.sequences_parquet
    SEQUENCES_LEGACY_PARQUET = cfg.sequences_legacy_parquet
    SEQUENCES_WINDOWS_APT_PARQUET = cfg.sequences_windows_apt_parquet
    ENTERPRISE_ATTACK_JSON = cfg.enterprise_attack_json
    COUNTS_UNIGRAM_PARQUET = cfg.counts_unigram_parquet
    COUNTS_BIGRAM_PARQUET = cfg.counts_bigram_parquet
    COUNTS_TRIGRAM_PARQUET = cfg.counts_trigram_parquet
    TECH_EMBEDDINGS_NPY = cfg.tech_embeddings_npy
    TECH_INDEX_CSV = cfg.tech_index_csv
    WORD2VEC_MODEL = cfg.word2vec_model
    PATHS_TOP_K_JSON = cfg.paths_top_k_json
    METRICS_JSON = cfg.metrics_json
    RESULTS_NOTEBOOK = cfg.results_notebook
    TACTIC_MAPPING_JSON = cfg.tactic_mapping_json
    TACTIC_TRANSITIONS_JSON = cfg.tactic_transitions_json
    ONTOLOGY_JSON = cfg.ontology_json
    ONTOLOGY_RULES_JSON = cfg.ontology_rules_json
    CAUSAL_ANNOTATIONS_JSON = cfg.causal_annotations_json
    DEFAULT_PARAMS = cfg.default_params
    WORD2VEC_PARAMS = cfg.word2vec_params
    BEAM_SEARCH_PARAMS = cfg.beam_search_params
    EVAL_PARAMS = cfg.eval_params
    COMPONENT_WEIGHTS_JSON = cfg.component_weights_json
    MODEL_TYPES = cfg.model_types
    DEFAULT_MODEL_TYPE = cfg.default_model_type
    HMM_DEFAULT_CONFIG = cfg.hmm_default_config
    BILSTM_DEFAULT_CONFIG = cfg.bilstm_default_config
    LSTM_DEFAULT_CONFIG = cfg.lstm_default_config
    GRU_DEFAULT_CONFIG = cfg.gru_default_config
    TCN_DEFAULT_CONFIG = cfg.tcn_default_config
    TRANSFORMER_DEFAULT_CONFIG = cfg.transformer_default_config
    DATA_SOURCE_CHOICES = tuple(cfg.data_sources.get('choices', ("legacy", "windows-apt", "both")))
    DEFAULT_DATA_SOURCE = cfg.data_sources.get('default', 'legacy')
    SEQUENCE_OUTPUT_MAP = {
        'legacy': cfg.sequences_legacy_parquet,
        'windows-apt': cfg.sequences_windows_apt_parquet,
        'both': cfg.sequences_parquet,
    }


def normalize_data_source(value: Optional[str]) -> str:
    """Normalize user input into a known data source identifier."""

    if value is None:
        return DEFAULT_DATA_SOURCE
    normalized = value.strip().lower().replace('_', '-')
    if normalized not in DATA_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported data source '{value}'. Expected one of {DATA_SOURCE_CHOICES}."
        )
    return normalized


def iter_data_sources(selection: Optional[str]) -> Iterable[str]:
    """Expand a selector such as "both" into explicit data sources."""

    normalized = normalize_data_source(selection)
    if normalized == 'both':
        return ('legacy', 'windows-apt')
    return (normalized,)


_refresh_exports()
