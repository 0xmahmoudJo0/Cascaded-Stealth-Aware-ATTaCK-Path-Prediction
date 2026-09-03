"""
Command-line interface for the chain-aware predictor pipeline.

This module provides CLI commands to run the complete pipeline or individual steps.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import pandas as pd

from . import config as cfg
from .sigma_stealth import build_sigma_stealth_csv
from .preprocess import (
    preprocess_sightings,
)
from .counts import compute_all_counts
from .embeddings import train_embeddings
from .beam_search import run_beam_search
from .predictor import load_predictor
from .eval import run_evaluation
from .model_fitting import fit_component_weights
from .comparison import ModelComparator
from .models.checkpoint import save_model_checkpoint, load_model_checkpoint
from .lifecycle_controller import AttackLifecycleController
from .tactics import load_tactic_mapping, load_tactic_transitions
from .platforms import load_platform_mapping
from .ontology_reasoning import OntologyReasoner
from .io import (
    load_sequence_metadata_for_source,
    load_sequences_for_source,
    resolve_sequences_path,
)
from .log_utils import (
    log_duration,
    log_error,
    log_info,
    log_inputs,
    log_outputs,
    log_section,
    log_success,
    log_warning,
)


class ArtifactCache:
    """In-memory store for frequently accessed artifacts."""

    def __init__(self) -> None:
        self._sequences: Dict[str, List[List[str]]] = {}
        self._metadata: Dict[str, pd.DataFrame] = {}

    def sequences(self, data_source: str, loader: Callable[[], List[List[str]]]) -> List[List[str]]:
        if data_source not in self._sequences:
            self._sequences[data_source] = loader()
        return self._sequences[data_source]

    def metadata(self, data_source: str, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if data_source not in self._metadata:
            self._metadata[data_source] = loader()
        return self._metadata[data_source]

    def set_sequences(self, data_source: str, sequences: List[List[str]]) -> None:
        self._sequences[data_source] = sequences

    def set_metadata(self, data_source: str, metadata: pd.DataFrame) -> None:
        self._metadata[data_source] = metadata

    def invalidate(self, data_source: Optional[str] = None) -> None:
        if data_source is None:
            self._sequences.clear()
            self._metadata.clear()
        else:
            self._sequences.pop(data_source, None)
            self._metadata.pop(data_source, None)


_ARTIFACT_CACHE = ArtifactCache()


def _load_sequences_cached(data_source: str) -> List[List[str]]:
    return _ARTIFACT_CACHE.sequences(
        data_source,
        lambda: load_sequences_for_source(data_source),
    )


def _load_metadata_cached(data_source: str) -> pd.DataFrame:
    return _ARTIFACT_CACHE.metadata(
        data_source,
        lambda: load_sequence_metadata_for_source(data_source),
    )


def _seed_cache(data_source: str, sequences: List[List[str]], metadata: pd.DataFrame) -> None:
    _ARTIFACT_CACHE.set_sequences(data_source, sequences)
    _ARTIFACT_CACHE.set_metadata(data_source, metadata)


def _get_data_source(args: argparse.Namespace) -> str:
    """Extract the normalized data source from parsed arguments."""

    return getattr(args, 'data_source', cfg.DEFAULT_DATA_SOURCE)


def _add_data_source_argument(parser: argparse.ArgumentParser, *, default: Optional[str] = None) -> None:
    """Attach the ``--data-source`` option to a subcommand parser."""

    parser.add_argument(
        '--data-source',
        choices=cfg.DATA_SOURCE_CHOICES,
        default=default or cfg.DEFAULT_DATA_SOURCE,
        help=(
            "Select which telemetry corpus to use. "
            "'legacy' loads the original MITRE sightings, 'windows-apt' loads the Windows "
            "APT scenarios, and 'both' merges them chronologically."
        ),
    )


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--config',
        type=str,
        help='Path to a YAML config overriding dataset locations and hyper-parameters.',
    )


def _resolve_params(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Extract predictor parameter overrides from CLI arguments."""
    overrides: Dict[str, Any] = {}

    if getattr(args, 'cascaded', False):
        overrides['scoring_mode'] = 'cascaded'
        overrides['tactic_prior_alpha'] = getattr(args, 'tactic_prior_alpha', None) or 0.5
        # Cascaded = tactic branching + stealth re-ranking as explicit gates.
        # The log-linear weights still control within-branch candidate scoring
        # and drive the factor contribution percentages reported in the paper.
        # Honour any explicit --w-tactic / --w-stealth passed alongside --cascaded.
        w_tactic = getattr(args, 'w_tactic', None)
        w_stealth = getattr(args, 'w_stealth', None)
        if w_tactic is not None:
            overrides['w_tactic'] = w_tactic
        if w_stealth is not None:
            overrides['w_stealth'] = w_stealth
    elif getattr(args, 'pure_model', False):
        overrides['w_tactic'] = 0.0
        overrides['w_stealth'] = 0.0
    else:
        w_tactic = getattr(args, 'w_tactic', None)
        w_stealth = getattr(args, 'w_stealth', None)
        if w_tactic is not None:
            overrides['w_tactic'] = w_tactic
        if w_stealth is not None:
            overrides['w_stealth'] = w_stealth

    return overrides if overrides else None


def cmd_sigma(_args):
    """Process Sigma rule repository into stealth scores and metadata."""

    log_section("Processing Sigma Rules")
    log_inputs({
        'Sigma rules directory': cfg.SIGMA_RULES_DIR,
    })
    with log_duration("Sigma stealth export"):
        build_sigma_stealth_csv(
            cfg.SIGMA_RULES_DIR,
            cfg.SIGMA_STEALTH_CSV,
            cfg.SIGMA_RULE_DETAILS_JSON,
        )

    log_outputs({
        'Stealth scores CSV': cfg.SIGMA_STEALTH_CSV,
        'Rule metadata JSON': cfg.SIGMA_RULE_DETAILS_JSON,
    })
    log_success("Sigma processing complete")


def cmd_preprocess(args):
    """Preprocess sightings data into canonical sequences and metadata."""

    log_section("Preprocessing Sightings")
    data_source = _get_data_source(args)
    log_info(f"Target data source: {data_source}")
    inputs: Dict[str, Path] = {'Legacy sightings CSV': cfg.SIGHTINGS_CSV}
    if data_source in {'windows-apt', 'both'}:
        inputs['Windows APT dataset'] = cfg.WINDOWS_APT_DATASET_DIR
    log_inputs(inputs)

    with log_duration("Canonicalizing telemetry"):
        sequences, metadata = preprocess_sightings(
            cfg.SIGHTINGS_CSV,
            cfg.SEQUENCES_PARQUET,
            data_source=data_source,
            windows_dataset_dir=cfg.WINDOWS_APT_DATASET_DIR,
        )

    _seed_cache(data_source, sequences, metadata)
    outputs = {
        'Combined sequences': cfg.SEQUENCES_PARQUET,
        'Selected sequences': resolve_sequences_path(data_source),
    }
    if cfg.TACTIC_TRANSITIONS_JSON.exists():
        outputs['Tactic transitions'] = cfg.TACTIC_TRANSITIONS_JSON
    log_outputs(outputs)

    if metadata is not None and 'start_time' in metadata.columns:
        span_start = metadata['start_time'].min()
        span_end = metadata['start_time'].max()
        span_msg = f" spanning {span_start} -> {span_end}"
    else:
        span_msg = ""

    log_success(
        "Preprocessing complete! "
        f"Generated {len(sequences)} sequences{span_msg}."
    )


def cmd_counts(args):
    """Compute and persist n-gram counts for the selected corpus."""

    log_section("Computing N-gram Counts")
    data_source = _get_data_source(args)
    log_inputs({'Sequences parquet': resolve_sequences_path(data_source)})
    sequences = _load_sequences_cached(data_source)
    with log_duration("N-gram counting"):
        compute_all_counts(sequences)
    log_outputs({
        'Unigram counts': cfg.COUNTS_UNIGRAM_PARQUET,
        'Bigram counts': cfg.COUNTS_BIGRAM_PARQUET,
        'Trigram counts': cfg.COUNTS_TRIGRAM_PARQUET,
    })
    log_success("Count computation complete")


def cmd_embeddings(args):
    """Train technique embeddings and ANN index for the selected corpus."""

    log_section("Training Embeddings")
    data_source = _get_data_source(args)
    log_inputs({'Sequences parquet': resolve_sequences_path(data_source)})
    sequences = _load_sequences_cached(data_source)
    
    # Split sequences to prevent data leakage
    train_ratio = getattr(args, 'train_ratio', 0.8)
    random_seed = getattr(args, 'seed', None)
    
    if train_ratio < 1.0:
        from .eval import Evaluator
        import random
        import numpy as np
        from . import config as cfg
        
        if random_seed is None:
            random_seed = cfg.DEFAULT_PARAMS.get('RANDOM_SEED', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        metadata = _load_metadata_cached(data_source)
        temp_evaluator = Evaluator(None, list(sequences), metadata)
        train_sequences, test_sequences = temp_evaluator.split_sequences(
            train_ratio=train_ratio,
            temporal_split=False,
            random_seed=random_seed
        )
        
        log_info(f"⚠️  IMPORTANT: Training embeddings only on TRAIN split to prevent data leakage")
        log_info(f"  Split: {len(train_sequences)} train, {len(test_sequences)} test (seed={random_seed})")
        sequences_to_use = train_sequences
    else:
        log_info("Using all sequences for embeddings (train_ratio=1.0)")
        sequences_to_use = sequences
    
    with log_duration("Word2Vec training"):
        train_embeddings(sequences_to_use)
    log_outputs({
        'Embeddings': cfg.TECH_EMBEDDINGS_NPY,
        'Technique index': cfg.TECH_INDEX_CSV,
    })
    log_success("Embedding training complete")



def cmd_train_eval(args):
    """Train the predictor, optionally fit weights, and run evaluation."""

    log_section("Training Predictor and Evaluation")
    data_source = _get_data_source(args)
    log_inputs({
        'Sequences': resolve_sequences_path(data_source),
        'Counts': cfg.COUNTS_UNIGRAM_PARQUET,
        'Embeddings': cfg.TECH_EMBEDDINGS_NPY,
    })

    model_type = getattr(args, 'model', None)
    predictor = load_predictor(model_type=model_type, data_source=data_source)

    sequences = _load_sequences_cached(data_source)
    metadata = _load_metadata_cached(data_source)

    if getattr(args, 'fit_weights', False):
        log_info("Learning log-linear combination weights before evaluation...")
        fit_component_weights(
            predictor,
            sequences,
            metadata=metadata,
            train_ratio=args.fit_train_ratio,
            top_n_candidates=args.fit_top_n,
            regularization=args.fit_regularization,
        )

    beam_width = args.beam_width or cfg.BEAM_SEARCH_PARAMS['beam_width']
    max_depth = args.L_max or cfg.BEAM_SEARCH_PARAMS['max_depth']
    top_k = args.top_k or cfg.BEAM_SEARCH_PARAMS['top_k_paths']

    if args.skip_beam_search and cfg.PATHS_TOP_K_JSON.exists():
        log_info(
            "Skipping beam search as requested and reusing "
            f"{cfg.PATHS_TOP_K_JSON}"
        )
    else:
        if args.skip_beam_search:
            log_warning(
                "--skip-beam-search provided but no cached paths exist; running search anyway."
            )
        log_info(
            f"Running beam search (width={beam_width}, depth={max_depth}, top_k={top_k})"
        )
        run_beam_search(
            predictor,
            cfg.PATHS_TOP_K_JSON,
            beam_width,
            max_depth,
            top_k,
        )

    results = run_evaluation(
        predictor,
        sequences,
        cfg.METRICS_JSON,
        args.skip_ablation,
        sequence_metadata=metadata,
    )
    _print_evaluation_summary(results)
    log_outputs({'Evaluation metrics': cfg.METRICS_JSON})
    log_success("Training and evaluation complete")


def _print_evaluation_summary(results: Dict[str, Any]):
    """Prints a formatted summary of the evaluation results."""
    print("\n" + "="*25)
    print("  Evaluation Summary")
    print("="*25)

    if not results:
        print("No evaluation results to display.")
        return

    _print_dataset_summary(results.get('dataset_info', {}))
    _print_next_step_summary(
        results.get('next_step_metrics', {}),
        results.get('next_step_confidence_intervals', {})
    )
    _print_path_level_summary(results.get('path_metrics', {}))
    _print_ablation_summary(results.get('ablation_results', {}))
    print("\n" + "="*25)

def _print_dataset_summary(info: Dict[str, Any]):
    if not info: return
    print("\n--- Dataset Info ---")
    print(f"  Total Sequences: {info.get('total_sequences', 'N/A')}")
    print(f"  Train Sequences: {info.get('train_sequences', 'N/A')}")
    print(f"  Test Sequences:  {info.get('test_sequences', 'N/A')}")
    print(f"  Avg. Seq. Length: {info.get('avg_sequence_length', 0.0):.2f}")

def _print_next_step_summary(metrics: Dict[str, Any], cis: Dict[str, Any]):
    if not metrics: return
    print("\n--- Next-Step Prediction Performance ---")
    mrr = metrics.get('MRR', {}).get('mean', 0.0)
    mrr_ci = cis.get('MRR', {})
    if mrr_ci:
        ci_str = f" (95% CI: [{mrr_ci.get('lower', 0.0):.4f}, {mrr_ci.get('upper', 0.0):.4f}])"
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}{ci_str}")
    else:
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

    prec_keys = sorted([k for k in metrics.get('precision_at_k', {})], key=int)
    for k in prec_keys:
        prec = metrics.get('precision_at_k', {}).get(k, {}).get('mean', 0.0)
        ci = cis.get(f'precision_at_{k}', {})
        if ci:
            ci_str = f" (95% CI: [{ci.get('lower', 0.0):.4f}, {ci.get('upper', 0.0):.4f}])"
            print(f"  Precision@{k:<18} {prec:.4f}{ci_str}")
        else:
            print(f"  Precision@{k:<18} {prec:.4f}")

def _print_path_level_summary(metrics: Dict[str, Any]):
    if not metrics: return
    print("\n--- Path-Level Prediction Performance ---")
    hit_keys = sorted([k for k in metrics if 'Hit@' in k], key=lambda x: int(x.split('@')[-1]))
    for k_str in hit_keys:
        print(f"  {k_str}: {metrics.get(k_str, 0.0):.4f}")

def _print_ablation_summary(ablation: Dict[str, Any]):
    if not ablation: return
    print("\n--- Ablation Study (Impact on MRR) ---")
    base = ablation.get('original', {})
    base_mrr = base.get('metrics', {}).get('MRR', {}).get('mean', 0.0)
    print(f"  Full Model MRR: {base_mrr:.4f}")
    for name, res in ablation.items():
        if name == 'original':
            continue
        mrr = res.get('metrics', {}).get('MRR', {}).get('mean', 0.0)
        diff = res.get('mrr_difference') or {}
        delta = diff.get('mean_diff', base_mrr - mrr)
        p_value = diff.get('p_value')
        if p_value is not None:
            print(f"  - Removing '{name.replace('_', ' ')}': {mrr:.4f} "
                  f"(Δ = {delta:+.4f}, p = {p_value:.4f})")
        else:
            print(f"  - Removing '{name.replace('_', ' ')}': {mrr:.4f} "
                  f"(Δ = {delta:+.4f})")

def cmd_fit_weights(args):
    """Standalone command to fit log-linear weights."""
    log_section("Fitting Log-linear Weights")

    data_source = _get_data_source(args)
    predictor = load_predictor(data_source=data_source)
    sequences = _load_sequences_cached(data_source)
    metadata = _load_metadata_cached(data_source)

    learned = fit_component_weights(
        predictor,
        sequences,
        metadata=metadata,
        train_ratio=args.train_ratio,
        top_n_candidates=args.top_n,
        regularization=args.regularization,
    )

    log_info("Learned weights:")
    for key, value in learned.items():
        log_info(f"  {key}: {value:.4f}")
    log_success("Weight fitting complete")


def cmd_predict(args):
    """Generate attack paths from custom seeds."""
    log_section("Generating Attack Paths")

    data_source = _get_data_source(args)

    if not args.seeds:
        log_error("The 'predict' command requires at least one seed technique.")
        log_info('Usage: python -m src.chain_paths.cli predict --seeds "T1059,T1105"')
        sys.exit(1)

    params = _resolve_params(args)
    
    # Load predictor (unified path for both checkpoint and normal loading)
    checkpoint_path = getattr(args, 'checkpoint', None)
    model_type = getattr(args, 'model_type', None)
    
    if checkpoint_path:
        # Use load_predictor which handles both full checkpoints and raw state_dict
        log_info(f"Loading predictor with checkpoint: {checkpoint_path}")
        try:
            predictor = load_predictor(
                params=params,
                data_source=data_source,
                model_type=model_type,
                checkpoint_path=str(checkpoint_path),
            )
            if model_type:
                log_info(f"Using model type: {model_type}")
        except Exception as exc:
            log_error(f"Failed to load checkpoint: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        predictor = load_predictor(
            params=params,
            data_source=data_source,
            model_type=model_type,
        )
        if model_type:
            log_info(f"Using base model type: {model_type}")

    # Extract parameters
    beam_width = args.beam_width or cfg.BEAM_SEARCH_PARAMS['beam_width']
    max_depth = args.L_max or cfg.BEAM_SEARCH_PARAMS['max_depth']
    max_paths = getattr(args, 'max_paths', 10)
    num_tactic_branches = getattr(args, 'num_tactic_branches', None) or getattr(args, 'top_k_tactics', None) or 3
    techniques_per_branch = getattr(args, 'techniques_per_branch', 2)
    verbose = getattr(args, 'verbose', False)
    use_optimized = getattr(args, 'use_optimized', True)
    output_file = Path(args.output) if args.output else cfg.PATHS_TOP_K_JSON

    log_inputs({'Sequences': resolve_sequences_path(data_source)})
    log_info(f"Seed techniques: {args.seeds}")
    log_info(f"Tactic filtering enabled: {predictor.params.get('use_tactic_filtering', True)}")
    
    # Check if user wants factor breakdown in output
    show_factors = getattr(args, 'show_factors', False)
    if show_factors:
        log_info("Factor breakdown enabled - output will include LSTM/Tactic/Stealth scores")
    
    # Use optimized predictor for cleaner output with probability tracking
    if use_optimized:
        from .optimized_predictor import OptimizedPredictor, TacticBranchConfig
        
        config = TacticBranchConfig(
            num_tactic_branches=num_tactic_branches,
            techniques_per_branch=techniques_per_branch,
            max_depth=max_depth,
            max_paths=max_paths,
        )
        
        log_info(f"Using optimized predictor (tactic_branches={config.num_tactic_branches}, "
                 f"techniques_per_branch={config.techniques_per_branch}, depth={config.max_depth})")
        
        # Parse seed techniques from string format
        seed_list = [t.strip() for t in args.seeds.split(',') if t.strip()]
        
        scoring_mode = predictor.params.get('scoring_mode', 'log_linear')
        tactic_prior_alpha = predictor.params.get('tactic_prior_alpha', 1.0)
        opt_predictor = OptimizedPredictor(predictor, config, verbose=verbose, scoring_mode=scoring_mode, tactic_prior_alpha=tactic_prior_alpha)
        path_results = opt_predictor.generate_paths_with_tactic_branching(seed_list)
        
        # Convert results to output format with full probability details
        output_payload = {
            'seeds': args.seeds,
            'config': {
                'num_tactic_branches': config.num_tactic_branches,
                'techniques_per_branch': config.techniques_per_branch,
                'max_depth': config.max_depth,
                'max_paths': config.max_paths,
            },
            'paths': []
        }
        
        for i, result in enumerate(path_results, 1):
            path_entry = {
                'rank': i,
                'techniques': result.techniques,
                'tactic_sequence': result.tactic_sequence,
                'path_probability': result.path_probability,
                'log_probability': result.log_probability,
                'tactic_transition_probs': result.tactic_transition_probs,
                'dominant_tactic': result.dominant_tactic,
                'steps': []
            }
            
            for step in result.step_probabilities:
                step_entry = {
                    'position': step.position,
                    'technique': step.technique,
                    'tactic': step.tactic,
                    'prev_tactic': step.prev_tactic,
                    'probabilities': {
                        'combined': step.p_combined,
                        'within_tactic': step.p_within_tactic,
                        'bilstm': step.p_bilstm,
                        'tactic_transition': step.p_tactic,
                        'stealth': step.p_stealth,
                    },
                }
                
                # Add factor breakdown if requested
                if show_factors:
                    # Get weights from predictor params (handle both lstm and bilstm)
                    base_model_name = predictor.base_model.get_model_name() if predictor.base_model else 'ngram'
                    w_model = predictor.params.get(f'w_{base_model_name}', predictor.params.get('w_bilstm', 1.0))
                    w_tactic = predictor.params.get('w_tactic', 0.1)
                    w_stealth = predictor.params.get('w_stealth', 0.1)
                    
                    # Calculate WEIGHTED factor contributions (as used in log-linear model)
                    import math
                    
                    # Only include factors with non-zero weights
                    if w_model > 0:
                        contrib_model = math.exp(w_model * math.log(max(step.p_bilstm, 1e-10)))
                    else:
                        contrib_model = 0.0
                        
                    if w_tactic > 0:
                        contrib_tactic = math.exp(w_tactic * math.log(max(step.p_tactic, 1e-10)))
                    else:
                        contrib_tactic = 0.0
                        
                    if w_stealth > 0:
                        contrib_stealth = math.exp(w_stealth * math.log(max(step.p_stealth, 1e-10)))
                    else:
                        contrib_stealth = 0.0
                    
                    total = contrib_model + contrib_tactic + contrib_stealth
                    if total > 0:
                        step_entry['factor_contributions'] = {
                            'lstm': contrib_model / total,
                            'tactic': contrib_tactic / total,
                            'stealth': contrib_stealth / total,
                        }
                        # Also include raw scores and weights
                        step_entry['factor_scores'] = {
                            'lstm': step.p_bilstm,
                            'tactic': step.p_tactic,
                            'stealth': step.p_stealth,
                        }
                        step_entry['factor_weights'] = {
                            'lstm': w_model,
                            'tactic': w_tactic,
                            'stealth': w_stealth,
                        }
                
                path_entry['steps'].append(step_entry)
            
            # Compute rank-percentile path confidence
            n_seed = len(seed_list)
            mean_pct, confidence_steps = opt_predictor.compute_path_confidence(
                result.techniques, n_seed=n_seed
            )
            path_entry['path_confidence'] = mean_pct
            path_entry['confidence_details'] = confidence_steps

            output_payload['paths'].append(path_entry)
            
            # Log path summary
            log_info(f"Path {i}: {result.techniques} (prob={result.path_probability:.6f}, confidence={mean_pct:.1f}%)")
        
        output_file.write_text(json.dumps(output_payload, indent=2), encoding='utf-8')
        
    else:
        # Legacy path using AttackLifecycleController
        log_info(f"Running staged pipeline (branch_factor={beam_width}, depth={max_depth}, "
                 f"max_paths={max_paths}, top_k_tactics={num_tactic_branches})")
        
        controller = AttackLifecycleController(
            predictor, 
            branch_factor=beam_width, 
            max_depth=max_depth,
            max_paths=max_paths,
            top_k_tactics=num_tactic_branches,
            verbose=verbose,
        )
        results = controller.simulate_attack(args.seeds)

        for branch_id, paths in results.items():
            for path_idx, path_data in enumerate(paths, start=1):
                techniques = path_data.get("techniques") or []
                log_info(f"Branch {branch_id} path {path_idx}: {techniques}")

        output_payload = {
            'seeds': args.seeds,
            'branch_factor': beam_width,
            'max_depth': max_depth,
            'paths': results,
        }
        output_file.write_text(json.dumps(output_payload, indent=2), encoding='utf-8')

    log_outputs({'Generated paths': output_file})
    log_success("Path generation complete")

    # Optional: evaluate predicted paths against test sequences
    if getattr(args, 'evaluate_paths', False):
        from .path_eval import evaluate_predicted_paths
        from .tactics import load_tactic_mapping
        
        log_info("\n--- Evaluating predicted paths against test sequences ---")
        
        # Load test sequences
        sequences = _load_sequences_cached(data_source)
        n_train = int(len(sequences) * 0.8)
        test_sequences = sequences[n_train:]
        
        # Load tactic mapping
        tactic_mapping = load_tactic_mapping(cfg.TACTIC_MAPPING_JSON)
        
        # Run path evaluation
        path_metrics_file = output_file.parent / f"{output_file.stem}_path_metrics.json"
        path_report = evaluate_predicted_paths(
            predicted_paths_file=output_file,
            test_sequences=test_sequences,
            technique_to_tactics=predictor.technique_to_tactics,
            output_file=path_metrics_file,
            j_values=[1, 2, 3, 5],
            k_values=[1, 3, 5, 10],
        )
        
        log_outputs({'Path evaluation metrics': path_metrics_file})
        
        # Print summary
        if 'path_evaluation' in path_report:
            tactic_analysis = path_report['path_evaluation'].get('tactic_phase_analysis', {})
            if 'top_phases' in tactic_analysis and tactic_analysis['top_phases']:
                log_info("\nTop-2 Tactical Phases:")
                for i, phase in enumerate(tactic_analysis['top_phases'][:2], 1):
                    log_info(f"  {i}. {phase['phase_pattern']} (count={phase['count']}, {phase['fraction']:.1%})")

    # Optional: compute metrics on a sampled test set
    if getattr(args, 'metrics', False):
        from .eval import Evaluator
        # Load sequences and metadata
        sequences = _load_sequences_cached(data_source)
        metadata = _load_metadata_cached(data_source)

        # Use held-out 20% as test split
        n_train = int(len(sequences) * 0.8)
        test_sequences = sequences[n_train:]
        test_metadata = metadata.iloc[n_train:].reset_index(drop=True) if metadata is not None else None

        # Sample for speed
        sample_size = getattr(args, 'metrics_sample_size', 1000)
        if len(test_sequences) > sample_size:
            import random
            random.seed(42)
            indices = random.sample(range(len(test_sequences)), sample_size)
            test_sequences = [test_sequences[i] for i in indices]
            if test_metadata is not None:
                test_metadata = test_metadata.iloc[indices].reset_index(drop=True)

        print("\n--- Computing evaluation metrics on sampled test set ---")
        evaluator = Evaluator(predictor, test_sequences, sequence_metadata=test_metadata)
        metrics = evaluator.evaluate_all(train_ratio=0.0, temporal_split=False)

        # Save metrics
        metrics_path = output_file.parent / f"{output_file.stem}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        log_outputs({'Evaluation metrics': metrics_path})


def cmd_evaluate(args):
    """Evaluate a trained model checkpoint on test sequences."""
    log_section("Model Evaluation")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log_error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    data_source = _get_data_source(args)
    
    # Load checkpoint
    log_info(f"Loading checkpoint: {checkpoint_path}")
    try:
        from .embeddings import load_embeddings_and_index
        from .io import load_parquet, load_csv
        from .predictor import Predictor
        from .tactics import load_tactic_mapping, load_tactic_transitions
        from .platforms import load_platform_mapping
        from .ontology_reasoning import OntologyReasoner
        from .models.checkpoint import load_model_checkpoint
        import pickle
        
        # Load raw checkpoint to extract split info
        with open(checkpoint_path, 'rb') as f:
            checkpoint_dict = pickle.load(f)
        checkpoint_metadata = checkpoint_dict.get('metadata', {})
        split_info = checkpoint_metadata.get('split_info', None)
        
        if split_info:
            log_info(f"Found split info in checkpoint:")
            log_info(f"  Seed: {split_info.get('random_seed')}, Temporal: {split_info.get('temporal_split')}")
            log_info(f"  Train: {split_info.get('n_train')}, Test: {split_info.get('n_test')}")
        else:
            log_info("⚠️  No split info in checkpoint - will use manual split")
        
        embeddings_path = cfg.OUTPUTS_DIR / "tech_embeddings.npy"
        tech_index_path = cfg.OUTPUTS_DIR / "tech_index.csv"
        
        embeddings, tech_to_idx, _ = load_embeddings_and_index(
            embeddings_path,
            cfg.OUTPUTS_DIR / "emb_index.faiss",
            tech_index_path,
        )
        
        base_model = load_model_checkpoint(
            checkpoint_path,
            embeddings=embeddings,
            tech_to_idx=tech_to_idx,
        )
        
        # Load auxiliary artifacts (counts intentionally skipped — pure vocabulary scoring)
        counts_data = {}
        sigma_scores = {}
        if cfg.SIGMA_STEALTH_CSV.exists():
            sigma_df = load_csv(cfg.SIGMA_STEALTH_CSV)
            # Handle both 'technique_id' and 'tech_id' column names
            tech_col = 'technique_id' if 'technique_id' in sigma_df.columns else 'tech_id'
            sigma_scores = dict(zip(sigma_df[tech_col], sigma_df['stealth_S']))
        
        tactic_mapping = load_tactic_mapping(cfg.TACTIC_MAPPING_JSON)
        tactic_transitions = load_tactic_transitions(cfg.TACTIC_TRANSITIONS_JSON)
        platform_mapping = load_platform_mapping(cfg.ENTERPRISE_ATTACK_JSON)
        
        ontology_reasoner = None
        try:
            if cfg.ONTOLOGY_JSON.exists():
                ontology_reasoner = OntologyReasoner.from_files(
                    cfg.ONTOLOGY_JSON, cfg.ONTOLOGY_RULES_JSON
                )
        except Exception:
            pass
        
        predictor = Predictor(
            counts_data=counts_data,
            ps_scores={},
            sigma_scores=sigma_scores,
            embeddings=embeddings,
            tech_to_idx=tech_to_idx,
            index=None,
            base_model=base_model,
            tactic_mapping=tactic_mapping,
            tactic_transitions=tactic_transitions,
            platform_mapping=platform_mapping,
        )
        
        # Store checkpoint path for parallel workers
        predictor._checkpoint_path = str(checkpoint_path.resolve())
        
        log_success("Checkpoint loaded successfully")
        
    except Exception as exc:
        log_error(f"Failed to load checkpoint: {exc}")
        sys.exit(1)
    
    # Load test sequences
    log_info("Loading test sequences...")
    sequences = _load_sequences_cached(data_source)
    metadata = _load_metadata_cached(data_source)
    log_info(f"✓ Loaded {len(sequences)} total sequences")
    
    # Use saved split indices if available, otherwise fall back to manual split
    if split_info and 'test_indices' in split_info:
        test_indices = split_info['test_indices']
        test_sequences = [sequences[i] for i in test_indices if i < len(sequences)]
        test_metadata = metadata.iloc[test_indices].reset_index(drop=True) if metadata is not None else None
        log_info(f"✓ Using saved test split from checkpoint: {len(test_sequences)} sequences")
        log_info(f"  (Reproducible evaluation with seed={split_info.get('random_seed')})")
    else:
        # Fall back to manual split (backward compatibility)
        log_info("⚠️  No saved split indices - using manual split (not reproducible)")
        train_ratio = getattr(args, 'train_ratio', 0.8)
        n_train = int(len(sequences) * train_ratio)
        test_sequences = sequences[n_train:]
        test_metadata = metadata.iloc[n_train:].reset_index(drop=True) if metadata is not None else None
        log_info(f"✓ Train/test split: {n_train} train, {len(test_sequences)} test (ratio={train_ratio})")
    
    # Log sequence length distribution
    seq_lengths = [len(seq) for seq in test_sequences]
    if seq_lengths:
        log_info(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")
    
    # Apply seed filtering if requested
    eval_seeds = getattr(args, 'eval_seeds', None)
    eval_seed_mode = getattr(args, 'eval_seed_mode', 'prefix')
    
    if eval_seeds:
        log_info(f"Applying seed filter: {eval_seeds} (mode: {eval_seed_mode})")
        # Normalize seed list (handle both T1566_001 and T1566.001 formats)
        seed_list = []
        for s in eval_seeds.split(','):
            normalized = s.strip().upper().replace('_', '.')
            seed_list.append(normalized)
        log_info(f"  Normalized seeds: {seed_list}")
        
        filtered_sequences = []
        for seq in test_sequences:
            if not seq:
                continue
            
            # Normalize first technique in sequence
            first_tech = seq[0].upper().replace('_', '.')
            
            if eval_seed_mode == 'prefix' and first_tech in seed_list:
                filtered_sequences.append(seq)
            elif eval_seed_mode == 'contains':
                seq_normalized = [t.upper().replace('_', '.') for t in seq]
                if any(s in seed_list for s in seq_normalized):
                    filtered_sequences.append(seq)
        
        log_info(f"✓ Seed filter result: {len(test_sequences)} -> {len(filtered_sequences)} sequences")
        if len(filtered_sequences) == 0:
            # Show available seeds
            available_seeds = {}
            for seq in test_sequences:
                if seq:
                    first_tech = seq[0].upper().replace('_', '.')
                    available_seeds[first_tech] = available_seeds.get(first_tech, 0) + 1
            
            top_seeds = sorted(available_seeds.items(), key=lambda x: x[1], reverse=True)[:20]
            log_error("⚠️  SEED FILTER PRODUCED EMPTY TEST SET!")
            log_info(f"   Sample test sequences before filter: {test_sequences[:5]}")
            log_info(f"   Looking for seeds: {seed_list}")
            log_info(f"   Available starting techniques (top 20): {', '.join(f'{t}({n})' for t, n in top_seeds)}")
            log_info(f"   Suggestion: Use one of the available techniques as --eval-seeds")
        test_sequences = filtered_sequences
        
        if test_metadata is not None and len(test_sequences) < len(test_metadata):
            # Re-index metadata to match filtered sequences
            test_metadata = test_metadata.iloc[:len(test_sequences)].reset_index(drop=True)
    
    # Sample if requested
    sample_size = getattr(args, 'metrics_sample_size', None)
    if sample_size and len(test_sequences) > sample_size:
        import random
        random.seed(42)
        indices = random.sample(range(len(test_sequences)), sample_size)
        test_sequences = [test_sequences[i] for i in indices]
        if test_metadata is not None:
            test_metadata = test_metadata.iloc[indices].reset_index(drop=True)
        log_info(f"✓ Sampled {sample_size} sequences for evaluation")
    
    log_info(f"Final test set size: {len(test_sequences)} sequences")
    if not test_sequences:
        log_error("⚠️  TEST SET IS EMPTY - No sequences to evaluate!")
        sys.exit(1)
    
    # Run evaluation
    n_workers = getattr(args, 'n_workers', None)
    if n_workers:
        log_info(f"Evaluating on {len(test_sequences)} test sequences (n_workers={n_workers})...")
    else:
        log_info(f"Evaluating on {len(test_sequences)} test sequences (auto workers)...")
    from .eval import Evaluator, generate_plotting_csvs
    
    skip_ablation = getattr(args, 'skip_ablation', False)
    skip_prior_comparison = getattr(args, 'skip_prior_comparison', False)
    skip_path_level = getattr(args, 'skip_path_level', False)
    pure_model = getattr(args, 'pure_model', False)
    
    if skip_ablation:
        log_info("Skipping ablation study (--skip-ablation)")
    if skip_prior_comparison:
        log_info("Skipping prior comparison (--skip-prior-comparison)")
    if skip_path_level:
        log_info("Skipping path-level evaluation (--skip-path-level)")
    
    # Apply pure model mode: disable all fusion components for academic evaluation
    if pure_model:
        log_section("PURE MODEL MODE: Academic Evaluation")
        log_info("Disabling all fusion components to isolate neural model performance")
        predictor.params.update({
            'w_tactic': 0.0,
            'w_stealth': 0.0,
            'use_tactic_filtering': False,
            'use_platform_filtering': False,
        })
        log_info(f"  Model weight: w_bilstm={predictor.params.get('w_bilstm', predictor.params.get('w_lstm', predictor.params.get('w_gru', 1.0)))}")
        log_info(f"  Tactic weight: w_tactic={predictor.params['w_tactic']} (disabled)")
        log_info(f"  Stealth weight: w_stealth={predictor.params['w_stealth']} (disabled)")
        log_info(f"  Tactic filtering: {predictor.params['use_tactic_filtering']} (disabled)")
        log_info(f"  Platform filtering: {predictor.params['use_platform_filtering']} (disabled)")
        log_info("✓ Pure model evaluation: Only learned sequence patterns will be used")
    
    evaluator = Evaluator(predictor, test_sequences, sequence_metadata=test_metadata)
    metrics = evaluator.evaluate_all(
        train_ratio=0.0,
        temporal_split=False,
        n_workers=n_workers,
        skip_ablation=skip_ablation,
        skip_prior_comparison=skip_prior_comparison,
        skip_path_level=skip_path_level
    )
    
    # Save metrics JSON
    output_file = Path(args.output) if args.output else cfg.OUTPUTS_DIR / "evaluation_metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    
    # Generate plotting CSV files
    log_info("Generating plotting CSV files...")
    plot_dir = output_file.parent / "evaluation_plots"
    csv_files = generate_plotting_csvs(
        evaluator=evaluator,
        test_sequences=test_sequences,
        test_contexts=[None] * len(test_sequences),  # Use metadata contexts if needed
        k_values=[1, 3, 5, 10, 20, 50],
        output_dir=str(plot_dir),
        use_scenario_priors=None  # Use default from predictor
    )
    
    # Print summary
    log_section("EVALUATION RESULTS")
    
    # Next-step metrics
    if 'next_step_metrics' in metrics:
        nsm = metrics['next_step_metrics']
        # Access correct nested structure: nsm['model']['MRR'] not nsm['MRR']
        model_metrics = nsm.get('model', {})
        mrr = model_metrics.get('MRR', {}).get('mean', 0.0)
        acc = model_metrics.get('accuracy', {}).get('mean', None)
        prec_at_k = model_metrics.get('precision_at_k', {})
        p1 = prec_at_k.get(1, prec_at_k.get('1', {})).get('mean', 0.0)
        p5 = prec_at_k.get(5, prec_at_k.get('5', {})).get('mean', 0.0)
        p10 = prec_at_k.get(10, prec_at_k.get('10', {})).get('mean', 0.0)
        
        log_info("─── Next-Step Prediction ───")
        log_info(f"  MRR:          {mrr:.4f}")
        # Prefer explicit accuracy if present; otherwise alias to Precision@1
        log_info(f"  Accuracy:     {(acc if acc is not None else p1):.4f}")
        log_info(f"  Precision@1:  {p1:.4f}")
        log_info(f"  Precision@5:  {p5:.4f}")
        log_info(f"  Precision@10: {p10:.4f}")
    
    # Path-level metrics
    if 'path_metrics' in metrics and metrics['path_metrics']:
        pm = metrics['path_metrics']
        log_info("─── Path-Level Prediction ───")
        for k in [5, 10, 20]:
            hit_k = pm.get(f'Hit@{k}', 0.0)
            log_info(f"  Hit@{k}:       {hit_k:.4f}")

    # Path probability metrics (full ground-truth paths)
    if 'path_probability_metrics' in metrics and metrics['path_probability_metrics']:
        ppm = metrics['path_probability_metrics']
        log_info("─── Path Probability ───")
        log_info(f"  Paths:        {ppm.get('paths_evaluated', 0)}")
        log_info(f"  Steps:        {ppm.get('total_steps', 0)} (zero-prob: {ppm.get('zero_prob_steps', 0)})")
        log_info(f"  Mean avg logP:{ppm.get('mean_avg_log_prob', 0.0):.6f}")
        log_info(f"  Mean perplex.:{ppm.get('mean_perplexity', 0.0):.4f}")
    
    log_outputs({'Evaluation metrics': output_file})
    log_outputs({'Plotting CSVs': plot_dir})
    log_success("Evaluation complete")


def _train_and_save_single(model_type: str, args) -> None:
    """Shared training helper for single-model CLI commands."""

    log_section(f"{model_type.upper()} Model Training")

    data_source = _get_data_source(args)
    sequences = _load_sequences_cached(data_source)
    metadata = _load_metadata_cached(data_source)

    log_info(f"Loaded {len(sequences)} sequences from {data_source}")

    comparator = ModelComparator()
    
    # Get random seed from args or config
    random_seed = getattr(args, 'seed', None)

    trained_models, split_info = comparator.train_all_models(
        sequences=sequences,
        metadata=metadata,
        train_ratio=args.train_ratio,
        model_types=[model_type],
        fast_train=getattr(args, 'fast_train', None),
        random_seed=random_seed,
        batch_size=getattr(args, 'batch_size', 32),
        epochs=getattr(args, 'epochs', 10),
        learning_rate=getattr(args, 'lr', 3e-4),
        num_workers=getattr(args, 'num_workers', 0),
        early_stopping=getattr(args, 'early_stopping', False),
        patience=getattr(args, 'patience', 5),
        min_delta=getattr(args, 'min_delta', 0.0001),
        validation_split=getattr(args, 'validation_split', 0.15),
        restore_best_weights=not getattr(args, 'no_restore_best', False),
        early_stopping_metric=getattr(args, 'es_metric', 'loss'),
    )

    if not trained_models or model_type not in trained_models:
        log_error(f"Failed to train {model_type} model. Check error messages above.")
        sys.exit(1)

    model = trained_models[model_type]

    # Determine save path
    default_save = cfg.OUTPUTS_DIR / f"{model_type}.pkl"
    save_arg = getattr(args, 'save_model', None)
    save_path = Path(save_arg) if save_arg else default_save

    try:
        checkpoint_metadata = {
            'epochs': getattr(args, 'epochs', None),
            'batch_size': getattr(args, 'batch_size', None),
            'train_ratio': getattr(args, 'train_ratio', None),
            'fast_train': getattr(args, 'fast_train', None),
            'early_stopping': getattr(args, 'early_stopping', False),
            'patience': getattr(args, 'patience', None),
            'min_delta': getattr(args, 'min_delta', None),
            'validation_split': getattr(args, 'validation_split', None),
            'early_stopping_metric': getattr(args, 'es_metric', None),
            # Add split information for reproducibility
            'split_info': split_info,
        }
        save_model_checkpoint(
            model=model,
            path=save_path,
            model_type=model_type,
            metadata=checkpoint_metadata,
        )
        log_success(f"Saved {model_type} model to {save_path}")
        log_info(f"  Split: {split_info['n_train']} train, {split_info['n_test']} test (seed={split_info['random_seed']})")
    except Exception as exc:  # pragma: no cover - IO errors
        log_error(f"Failed to save model to {save_path}: {exc}")
        sys.exit(1)

    log_success(f"{model_type.upper()} training complete")


def cmd_train_bilstm(args):
    """Train BiLSTM model alone with progress tracking and ETA."""
    _train_and_save_single('bilstm', args)


def cmd_train_lstm(args):
    """Train unidirectional LSTM model with progress tracking and ETA."""
    _train_and_save_single('lstm', args)


def cmd_train_ngram(args):
    """Train N-Gram model and save artifact."""
    _train_and_save_single('ngram', args)


def cmd_train_hmm(args):
    """Train HMM model and save artifact."""
    _train_and_save_single('hmm', args)


def cmd_train_gru(args):
    """Train GRU model and save artifact."""
    _train_and_save_single('gru', args)


def cmd_train_tcn(args):
    """Train TCN model and save artifact."""
    _train_and_save_single('tcn', args)


def cmd_train_transformer(args):
    """Train Transformer model and save artifact."""
    _train_and_save_single('transformer', args)


def cmd_compare_models(args):
    """Compare multiple models and select the best one."""
    log_section("Model Comparison")

    # Load sequences
    data_source = _get_data_source(args)
    sequences = _load_sequences_cached(data_source)
    metadata = _load_metadata_cached(data_source)
    
    # Parse model list
    if args.models:
        model_types = [m.strip() for m in args.models.split(',')]
    else:
        from .models.factory import ModelFactory
        model_types = ModelFactory.get_available_models()
    
    log_info(f"Comparing models: {', '.join(model_types)}")
    
    # Create comparator
    comparator = ModelComparator()
    
    # Train all models
    trained_models = comparator.train_all_models(
        sequences=sequences,
        metadata=metadata,
        train_ratio=args.train_ratio,
        model_types=model_types,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    
    if not trained_models:
        log_error("No models were successfully trained.")
        sys.exit(1)
    
    # Split test sequences
    n_train = int(len(sequences) * args.train_ratio)
    test_sequences = sequences[n_train:]
    test_metadata = metadata.iloc[n_train:].reset_index(drop=True) if metadata is not None else None
    
    # Evaluate all models
    results = comparator.evaluate_all_models(
        models=trained_models,
        test_sequences=test_sequences,
        metadata=test_metadata,
    )
    
    # Generate comparison
    output_file = Path(args.output) if args.output else cfg.OUTPUTS_DIR / "model_comparison.json"
    comparison = comparator.compare_results(results, output_file)
    
    # Select best model
    best_model = comparator.select_best_model(results, metric=args.metric)
    
    # Print summary
    log_section("COMPARISON SUMMARY")
    
    for model_name, model_data in comparison['models'].items():
        if 'error' in model_data:
            log_error(f"{model_name.upper()}: {model_data['error']}")
        else:
            log_info(f"{model_name.upper()}:")
            log_info(f"  MRR: {model_data['mrr']:.4f}")
            log_info(f"  Precision@1: {model_data['precision_at_1']:.4f}")
            log_info(f"  Precision@5: {model_data['precision_at_5']:.4f}")
            log_info(f"  Hit@5: {model_data['hit_at_5']:.4f}")

    if best_model:
        log_success(f"BEST MODEL (by {args.metric}): {best_model.upper()}")
        log_info(
            "To use this model, run: "
            f"python -m src.chain_paths.cli train_eval --model {best_model}"
        )

    log_outputs({'Comparison report': output_file})


def cmd_full_pipeline(args):
    """Run the complete pipeline."""
    log_section("Running Full Pipeline")

    data_source = _get_data_source(args)

    # Create a dummy args object for subcommand calls
    sub_args = argparse.Namespace(data_source=data_source, config=args.config)

    # Run steps in order
    cmd_sigma(sub_args)
    cmd_preprocess(sub_args)
    cmd_counts(sub_args)
    cmd_embeddings(sub_args)

    # For train_eval, we need to populate the args from the 'full' command's args
    # or use defaults. Since 'full' has no args, we can create a default set.
    train_eval_args = argparse.Namespace(
        data_source=data_source,
        fit_weights=False,
        skip_beam_search=False,
        skip_ablation=False,
        beam_width=None,
        L_max=None,
        top_k=None,
        model=None,
        fit_train_ratio=cfg.EVAL_PARAMS['train_test_split'],
        fit_top_n=50,
        fit_regularization=None,
        config=args.config,
    )
    cmd_train_eval(train_eval_args)

    log_success("Full pipeline complete")


def cmd_visualize(args):
    """Visualize attack path graphs with tactic branching."""
    log_section("Visualizing Attack Paths")
    
    from .visualize_attack_graph import visualize_from_cli
    
    input_path = Path(args.input)
    if not input_path.exists():
        log_error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / 'graphs'
    
    log_inputs({'Input paths': input_path})
    log_info(f"Output directory: {output_path}")
    log_info(f"Format: {args.format}, DPI: {args.dpi}")
    if args.max_paths:
        log_info(f"Max paths: {args.max_paths}")
    
    try:
        visualize_from_cli(
            input_path=input_path,
            output_path=output_path,
            max_paths=args.max_paths,
            show_probabilities=args.show_probabilities,
            output_format=args.format,
            dpi=args.dpi,
        )
        log_outputs({'Graphs': output_path})
        log_success("Visualization complete")
    except Exception as e:
        log_error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main(argv: Optional[List[str]] = None):
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Chain-aware attack path predictor")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    subparsers.required = True

    # Sigma command
    sigma_parser = subparsers.add_parser('sigma', help='Process Sigma rules')
    _add_config_argument(sigma_parser)

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess sightings data')
    _add_config_argument(preprocess_parser)
    _add_data_source_argument(preprocess_parser)

    # Counts command
    counts_parser = subparsers.add_parser('counts', help='Compute n-gram counts')
    _add_config_argument(counts_parser)
    _add_data_source_argument(counts_parser)

    # Embeddings command
    emb_parser = subparsers.add_parser('emb', help='Train technique embeddings')
    _add_config_argument(emb_parser)
    _add_data_source_argument(emb_parser)
    emb_parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of sequences for training (rest is test, to prevent data leakage)')
    emb_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible train/test split')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model on test sequences')
    _add_config_argument(evaluate_parser)
    _add_data_source_argument(evaluate_parser)
    evaluate_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pkl file)')
    evaluate_parser.add_argument('--output', type=str, help='Output JSON file for metrics')
    evaluate_parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of data used for training (rest is test)')
    evaluate_parser.add_argument('--random-state', type=int, default=42, help='Random seed for train/test split (default: 42)')
    evaluate_parser.add_argument('--shuffle', action='store_true', help='Shuffle data before splitting')
    evaluate_parser.add_argument('--eval-seeds', type=str, help='Comma-separated seed techniques to filter test sequences (e.g., "T1566_001,T1078")')
    evaluate_parser.add_argument('--eval-seed-mode', choices=['prefix', 'contains'], default='prefix', help='How to filter by seeds (default: prefix)')
    evaluate_parser.add_argument('--metrics-sample-size', type=int, help='Sample N sequences for faster evaluation')
    evaluate_parser.add_argument('--n-workers', type=int, default=None, help='Number of parallel workers (None=auto, 1=sequential, >1=parallel)')
    evaluate_parser.add_argument('--skip-ablation', action='store_true', help='Skip the ablation study (6 extra evaluation passes)')
    evaluate_parser.add_argument('--skip-prior-comparison', action='store_true', help='Skip prior variant comparison (saves 1 extra pass)')
    evaluate_parser.add_argument('--skip-path-level', action='store_true', help='Skip path-level beam search evaluation (very slow, 500+ beam searches)')
    evaluate_parser.add_argument('--pure-model', action='store_true', help='Evaluate pure model only (disable stealth and tactic fusion for academic evaluation)')

    # Train and evaluate command
    train_eval_parser = subparsers.add_parser('train_eval', help='Train predictor and run evaluation')
    _add_config_argument(train_eval_parser)
    _add_data_source_argument(train_eval_parser)
    train_eval_parser.add_argument('--model', choices=cfg.MODEL_TYPES, help='Base model family to use')
    train_eval_parser.add_argument('--fusion', choices=['off', 'loglinear', 'mlp'], help='Fusion strategy to apply')
    train_eval_parser.add_argument('--no-fusion', action='store_true', help='Disable the fusion layer regardless of configuration')
    train_eval_parser.add_argument('--beam_width', type=int, help='Beam width for search')
    train_eval_parser.add_argument('--l-max', type=int, dest='L_max', help='Maximum path depth')
    train_eval_parser.add_argument('--top_k', type=int, help='Number of top paths')
    train_eval_parser.add_argument(
        '--skip-beam-search', action='store_true',
        help='Skip beam search if output file exists'
    )
    train_eval_parser.add_argument(
        '--skip-ablation', action='store_true',
        help='Skip the ablation study during evaluation'
    )
    train_eval_parser.add_argument(
        '--fit-weights', action='store_true',
        help='Fit log-linear weights prior to evaluation'
    )
    train_eval_parser.add_argument(
        '--fit-train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'],
        help='Train ratio for weight fitting (default from eval config)'
    )
    train_eval_parser.add_argument(
        '--fit-top-n', type=int, default=50,
        help='Candidate count used during weight fitting'
    )
    train_eval_parser.add_argument(
        '--fit-regularization', type=float,
        help='L2 regularization strength for weight fitting (defaults to config value)'
    )

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate attack paths from custom seeds')
    _add_config_argument(predict_parser)
    _add_data_source_argument(predict_parser)
    predict_parser.add_argument(
        '--seeds', type=str, required=True,
        help='Custom seed techniques. Format: "T1566,T1204;T1059" (comma=sequence, semicolon=multiple seeds)'
    )
    predict_parser.add_argument(
        '--model-type', choices=cfg.MODEL_TYPES,
        help='Base model family to use (optional; auto-detected from checkpoint if omitted)'
    )
    predict_parser.add_argument(
        '--checkpoint', type=str,
        help='Path to a saved model artifact (auto-detects model type from .pkl file)'
    )
    predict_parser.add_argument('--output', type=str, help='Output JSON file for generated paths')
    predict_parser.add_argument('--beam-width', type=int, dest='beam_width', default=200, help='Beam width for search (default: 200)')
    predict_parser.add_argument('--max-depth', type=int, dest='L_max', default=6, help='Maximum path depth (default: 6)')
    predict_parser.add_argument('--max-paths', type=int, default=10, help='Maximum number of paths to generate (default: 10)')
    predict_parser.add_argument(
        '--num-tactic-branches', type=int, default=None, 
        help='Number of tactical directions to explore at each step '
             '(default: 3 for log-linear, max_paths for cascaded mode)'
    )
    predict_parser.add_argument(
        '--techniques-per-branch', type=int, default=2,
        help='Number of techniques to consider per tactic branch (default: 2)'
    )
    predict_parser.add_argument('--top-k-tactics', type=int, default=None, help='[Legacy] Alias for --num-tactic-branches')
    predict_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging during prediction')
    predict_parser.add_argument('--evaluate-paths', action='store_true', help='Evaluate predicted paths against test sequences')
    predict_parser.add_argument('--metrics', action='store_true', help='Compute evaluation metrics after path generation')
    predict_parser.add_argument('--metrics-sample-size', type=int, default=1000, help='Number of sequences to sample for metrics')
    predict_parser.add_argument('--use-optimized', action='store_true', default=True, help='Use optimized predictor (default: True)')
    predict_parser.add_argument('--no-optimized', action='store_false', dest='use_optimized', help='Use legacy predictor')
    predict_parser.add_argument(
        '--show-factors', action='store_true',
        help='Include factor breakdown (LSTM, Tactic, Stealth) in output JSON'
    )
    predict_parser.add_argument(
        '--w-tactic', type=float, default=None, dest='w_tactic',
        help='Override tactic fusion weight (default: 0.1 from config)'
    )
    predict_parser.add_argument(
        '--w-stealth', type=float, default=None, dest='w_stealth',
        help='Override stealth fusion weight (default: 0.1 from config)'
    )
    predict_parser.add_argument(
        '--pure-model', action='store_true', dest='pure_model',
        help='Disable tactic and stealth fusion (sets both weights to 0.0)'
    )
    predict_parser.add_argument(
        '--cascaded', action='store_true',
        help='Use cascaded scoring: LSTM generates, tactic filters, stealth selects (no weight tuning)'
    )
    predict_parser.add_argument(
        '--tactic-prior-alpha', type=float, dest='tactic_prior_alpha', default=None,
        help='Temperature for tactic-transition prior in cascaded mode (default: 0.5). '
             '1.0=full enforcement, 0.5=calibrated balance allowing minority paths'
    )
    predict_parser.set_defaults(func=cmd_predict)

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize attack path graphs with tactic branching')
    visualize_parser.add_argument('--input', type=str, required=True, help='Input JSON file from predict command')
    visualize_parser.add_argument('--output', type=str, help='Output directory for graphs (default: <input_dir>/graphs)')
    visualize_parser.add_argument('--max-paths', type=int, help='Maximum number of paths to visualize (default: all)')
    visualize_parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default='png', help='Output format (default: png)')
    visualize_parser.add_argument('--dpi', type=int, default=300, help='Image resolution (default: 300)')
    visualize_parser.add_argument('--no-probabilities', action='store_false', dest='show_probabilities', help='Hide probability labels')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    _add_config_argument(full_parser)
    _add_data_source_argument(full_parser)

    # Fit weights command
    fit_parser = subparsers.add_parser('fit_weights', help='Fit log-linear combination weights')
    _add_config_argument(fit_parser)
    _add_data_source_argument(fit_parser)
    fit_parser.add_argument(
        '--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'],
        help='Fraction of sequences used for fitting'
    )
    fit_parser.add_argument(
        '--top-n', type=int, default=50,
        help='Candidate budget per context during fitting'
    )
    fit_parser.add_argument(
        '--regularization', type=float,
        help='L2 regularization strength; defaults to config value when omitted'
    )

    # Train N-gram command
    ngram_parser = subparsers.add_parser('train_ngram', help='Train N-gram model and save artifact')
    _add_config_argument(ngram_parser)
    _add_data_source_argument(ngram_parser)
    ngram_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    ngram_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples (optional)')
    ngram_parser.add_argument('--save-model', type=str, default=None, help='Save path (default: outputs/ngram.pkl)')

    # Train HMM command
    hmm_parser = subparsers.add_parser('train_hmm', help='Train HMM model and save artifact')
    _add_config_argument(hmm_parser)
    _add_data_source_argument(hmm_parser)
    hmm_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    hmm_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples (optional)')
    hmm_parser.add_argument('--save-model', type=str, default=None, help='Save path (default: outputs/hmm.pkl)')

    # Train BiLSTM command
    bilstm_parser = subparsers.add_parser('train_bilstm', help='Train BiLSTM model with progress tracking')
    _add_config_argument(bilstm_parser)
    _add_data_source_argument(bilstm_parser)
    bilstm_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples for fast iteration (e.g., 10000)')
    bilstm_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    bilstm_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible train/test split')
    bilstm_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    bilstm_parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs (default: 10)')
    bilstm_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4)')
    bilstm_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation metrics')
    bilstm_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5)')
    bilstm_parser.add_argument('--min-delta', type=float, default=0.0001, help='Minimum improvement threshold for early stopping (default: 0.0001)')
    bilstm_parser.add_argument('--validation-split', type=float, default=0.15, help='Fraction of training data for validation (default: 0.15)')
    bilstm_parser.add_argument('--es-metric', type=str, default='loss', choices=['loss', 'mrr', 'f1', 'recall_at_k', 'precision_at_k'], help='Metric to monitor for early stopping (default: loss)')
    bilstm_parser.add_argument('--no-restore-best', action='store_true', help='Do not restore best weights after early stopping')
    bilstm_parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes for faster batching (default: 0)')
    bilstm_parser.add_argument('--save-model', '--checkpoint', type=str, default=None, help='Save path (default: outputs/bilstm.pkl)')
    bilstm_parser.set_defaults(func=cmd_train_bilstm)

    # Train LSTM command
    lstm_parser = subparsers.add_parser('train_lstm', help='Train unidirectional LSTM model with progress tracking')
    _add_config_argument(lstm_parser)
    _add_data_source_argument(lstm_parser)
    lstm_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples for fast iteration (e.g., 10000)')
    lstm_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    lstm_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible train/test split')
    lstm_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    lstm_parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs (default: 10)')
    lstm_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4)')
    lstm_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation metrics')
    lstm_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5)')
    lstm_parser.add_argument('--min-delta', type=float, default=0.0001, help='Minimum improvement threshold for early stopping (default: 0.0001)')
    lstm_parser.add_argument('--validation-split', type=float, default=0.15, help='Fraction of training data for validation (default: 0.15)')
    lstm_parser.add_argument('--es-metric', type=str, default='loss', choices=['loss', 'mrr', 'f1', 'recall_at_k', 'precision_at_k'], help='Metric to monitor for early stopping (default: loss)')
    lstm_parser.add_argument('--no-restore-best', action='store_true', help='Do not restore best weights after early stopping')
    lstm_parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes for faster batching (default: 0)')
    lstm_parser.add_argument('--save-model', '--checkpoint', type=str, default=None, help='Save path (default: outputs/lstm.pkl)')
    lstm_parser.set_defaults(func=cmd_train_lstm)

    # Train GRU command
    gru_parser = subparsers.add_parser('train_gru', help='Train GRU model and save artifact')
    _add_config_argument(gru_parser)
    _add_data_source_argument(gru_parser)
    gru_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples (optional)')
    gru_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    gru_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    gru_parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs (default: 10)')
    gru_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4)')
    gru_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation metrics')
    gru_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5)')
    gru_parser.add_argument('--min-delta', type=float, default=0.0001, help='Minimum improvement threshold for early stopping (default: 0.0001)')
    gru_parser.add_argument('--validation-split', type=float, default=0.15, help='Fraction of training data for validation (default: 0.15)')
    gru_parser.add_argument('--es-metric', type=str, default='loss', choices=['loss', 'mrr', 'f1', 'recall_at_k', 'precision_at_k'], help='Metric to monitor for early stopping (default: loss)')
    gru_parser.add_argument('--no-restore-best', action='store_true', help='Do not restore best weights after early stopping')
    gru_parser.add_argument('--save-model', type=str, default=None, help='Save path (default: outputs/gru.pkl)')

    # Train TCN command
    tcn_parser = subparsers.add_parser('train_tcn', help='Train TCN model and save artifact')
    _add_config_argument(tcn_parser)
    _add_data_source_argument(tcn_parser)
    tcn_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples (optional)')
    tcn_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    tcn_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    tcn_parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs (default: 10)')
    tcn_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4)')
    tcn_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation metrics')
    tcn_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5)')
    tcn_parser.add_argument('--min-delta', type=float, default=0.0001, help='Minimum improvement threshold for early stopping (default: 0.0001)')
    tcn_parser.add_argument('--validation-split', type=float, default=0.15, help='Fraction of training data for validation (default: 0.15)')
    tcn_parser.add_argument('--es-metric', type=str, default='loss', choices=['loss', 'mrr', 'f1', 'recall_at_k', 'precision_at_k'], help='Metric to monitor for early stopping (default: loss)')
    tcn_parser.add_argument('--no-restore-best', action='store_true', help='Do not restore best weights after early stopping')
    tcn_parser.add_argument('--save-model', type=str, default=None, help='Save path (default: outputs/tcn.pkl)')

    # Train Transformer command
    transformer_parser = subparsers.add_parser('train_transformer', help='Train Transformer model and save artifact')
    _add_config_argument(transformer_parser)
    _add_data_source_argument(transformer_parser)
    transformer_parser.add_argument('--fast-train', type=int, default=None, help='Limit to N training examples (optional)')
    transformer_parser.add_argument('--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'], help='Fraction of sequences used for training')
    transformer_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    transformer_parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs (default: 10)')
    transformer_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4)')
    transformer_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation metrics')
    transformer_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5)')
    transformer_parser.add_argument('--min-delta', type=float, default=0.0001, help='Minimum improvement threshold for early stopping (default: 0.0001)')
    transformer_parser.add_argument('--validation-split', type=float, default=0.15, help='Fraction of training data for validation (default: 0.15)')
    transformer_parser.add_argument('--es-metric', type=str, default='loss', choices=['loss', 'mrr', 'f1', 'recall_at_k', 'precision_at_k'], help='Metric to monitor for early stopping (default: loss)')
    transformer_parser.add_argument('--no-restore-best', action='store_true', help='Do not restore best weights after early stopping')
    transformer_parser.add_argument('--save-model', type=str, default=None, help='Save path (default: outputs/transformer.pkl)')

    # Compare models command
    compare_parser = subparsers.add_parser('compare_models', help='Train and compare multiple models')
    _add_config_argument(compare_parser)
    _add_data_source_argument(compare_parser)
    compare_parser.add_argument(
        '--models', type=str,
        help='Comma-separated list of models to compare (e.g., ngram,hmm,bilstm). Default: all models'
    )
    compare_parser.add_argument(
        '--metric', type=str, default='MRR',
        help='Metric to use for best model selection (MRR, precision_at_1, Hit@5, etc.)'
    )
    compare_parser.add_argument(
        '--train-ratio', type=float, default=cfg.EVAL_PARAMS['train_test_split'],
        help='Fraction of sequences used for training'
    )
    compare_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for neural model training'
    )
    compare_parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs for neural models'
    )
    compare_parser.add_argument(
        '--output', type=str,
        help='Output file for comparison results (default: outputs/model_comparison.json)'
    )

    # Parse arguments
    args = parser.parse_args(argv)

    # Load configuration overrides if provided
    cfg.configure_pipeline(getattr(args, 'config', None))

    # Execute command
    command_map = {
        'sigma': cmd_sigma, 'preprocess': cmd_preprocess, 'counts': cmd_counts,
        'emb': cmd_embeddings, 'train_eval': cmd_train_eval, 'evaluate': cmd_evaluate,
        'fit_weights': cmd_fit_weights, 'predict': cmd_predict, 'full': cmd_full_pipeline,
        'train_ngram': cmd_train_ngram, 'train_hmm': cmd_train_hmm,
        'train_bilstm': cmd_train_bilstm, 'train_lstm': cmd_train_lstm, 'train_gru': cmd_train_gru,
        'train_tcn': cmd_train_tcn, 'train_transformer': cmd_train_transformer,
        'compare_models': cmd_compare_models, 'visualize': cmd_visualize,
    }

    try:
        if args.command in command_map:
            command_map[args.command](args)
        else:
            log_error(f"Unknown command: {args.command}")
            sys.exit(1)

    except (IOError, ValueError, RuntimeError) as e:
        log_error(f"Error executing command '{args.command}': {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
