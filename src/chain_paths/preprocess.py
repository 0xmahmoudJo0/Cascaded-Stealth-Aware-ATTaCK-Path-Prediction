"""
Preprocessing module for sightings data.

This module canonicalizes technique IDs, builds sequences per campaign,
and computes n-gram counts for the chain-aware predictor.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as cfg
from .config import iter_data_sources, normalize_data_source
from .io import (
    WINDOWS_APT_COMBINED_FILENAME,
    ensure_dir,
    load_sightings_data,
    parse_windows_apt_sequences,
    save_parquet,
)
from .tactics import load_tactic_mapping, update_tactic_transitions_from_sequences
from .technique_utils import canonicalize_technique_id


def build_technique_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build mapping from raw technique strings to canonical TIDs.
    
    Args:
        df: Sightings DataFrame
        
    Returns:
        Dictionary mapping raw strings to canonical TIDs
    """
    print("Building technique ID mapping...")
    
    # Get unique technique strings
    if 'sequence' in df.columns:
        # Extract all techniques from sequences
        all_techniques = set()
        for sequence in df['sequence']:
            all_techniques.update(sequence)
        unique_techs = list(all_techniques)
    else:
        unique_techs = df['technique_id'].dropna().unique()
    
    mapping = {}
    canonical_counts = defaultdict(int)
    
    for tech_str in tqdm(unique_techs, desc="Canonicalizing techniques"):
        canonical = canonicalize_technique_id(tech_str)
        if canonical:
            mapping[tech_str] = canonical
            canonical_counts[canonical] += 1
    
    print(f"Canonicalized {len(mapping)} unique techniques")
    print(f"Resulting in {len(canonical_counts)} canonical TIDs")
    
    return mapping


def apply_technique_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply technique mapping to DataFrame.
    
    Args:
        df: Sightings DataFrame
        mapping: Technique mapping dictionary
        
    Returns:
        DataFrame with canonicalized technique IDs
    """
    print("Applying technique mapping...")
    
    # Create a copy to avoid modifying original
    df_mapped = df.copy()
    
    # Check if this is the v2 format with sequences
    if 'sequence' in df_mapped.columns:
        # Apply mapping to sequences
        def map_sequence(sequence):
            return [mapping.get(tech, tech) for tech in sequence]
        
        df_mapped['sequence'] = df_mapped['sequence'].apply(map_sequence)
        
        # Remove sequences with unmapped techniques (optional - we could keep them)
        before_count = len(df_mapped)
        # For now, keep all sequences
        after_count = len(df_mapped)
        
        print(f"Applied mapping to {after_count} sequences")
    else:
        # Apply mapping to individual techniques
        df_mapped['technique_id_canonical'] = df_mapped['technique_id'].map(mapping)
        
        # Remove rows with unmapped techniques
        before_count = len(df_mapped)
        df_mapped = df_mapped.dropna(subset=['technique_id_canonical'])
        after_count = len(df_mapped)
        
        print(f"Removed {before_count - after_count} rows with unmapped techniques")
        print(f"Retained {after_count} rows")
        
        # Rename column
        df_mapped = df_mapped.rename(columns={'technique_id_canonical': 'technique_id'})
    
    return df_mapped


def _derive_kill_chain_boundaries(tactic_sequence: Sequence[str]) -> List[Dict[str, int]]:
    """Convert a tactic sequence into contiguous phase boundaries."""

    if not tactic_sequence:
        return []

    boundaries: List[Dict[str, int]] = []
    current = tactic_sequence[0]
    start = 0

    for idx, tactic in enumerate(tactic_sequence[1:], start=1):
        if tactic != current:
            boundaries.append({'tactic': current, 'start': start, 'end': idx - 1})
            current = tactic
            start = idx

    boundaries.append({'tactic': current, 'start': start, 'end': len(tactic_sequence) - 1})
    return boundaries


def _resolve_tactic_sequence(
    sequence: Sequence[str],
    tactic_mapping: Mapping[str, Sequence[str]],
    missing_counter: Optional[Dict[str, int]] = None,
    skip_unmapped: bool = True,
) -> Optional[List[str]]:
    """Map each technique in ``sequence`` to a canonical tactic.
    
    Args:
        sequence: List of technique IDs
        tactic_mapping: Mapping from TID to list of tactics
        missing_counter: Optional counter for unmapped techniques
        skip_unmapped: If True, skip unmapped techniques. If False, drop entire sequence.
    
    Returns:
        List of tactics corresponding to mapped techniques, or None if sequence becomes empty
    """

    tactics: List[str] = []
    missing: List[str] = []

    for tech in sequence:
        tid = str(tech).strip().upper()
        options = tactic_mapping.get(tid)
        if not options:
            missing.append(tid)
            if not skip_unmapped:
                # Original behavior: drop on first unmapped
                if missing_counter is not None:
                    for m in missing:
                        missing_counter[m] = missing_counter.get(m, 0) + 1
                return None
            # New behavior: skip this technique, continue with others
            continue
        tactics.append(str(options[0]))

    # Track unmapped techniques
    if missing and missing_counter is not None:
        for tid in missing:
            missing_counter[tid] = missing_counter.get(tid, 0) + 1

    # Only drop if no techniques were mapped to tactics
    if len(tactics) == 0:
        return None

    return tactics


def build_sequences(
    df: pd.DataFrame,
    tactic_mapping: Optional[Mapping[str, Sequence[str]]] = None,
    skip_unmapped_techniques: bool = True,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Build technique sequences and capture metadata.

    Args:
        df: Canonicalized sightings DataFrame
        tactic_mapping: Mapping from TID to tactics
        skip_unmapped_techniques: If True, skip unmapped techniques instead of 
                                 dropping entire sequence

    Returns:
        Tuple containing the list of technique sequences and
        a DataFrame with sequence metadata such as sequence length.
    """
    print("Building technique sequences...")
    print(f"  Skip unmapped techniques: {skip_unmapped_techniques}")

    sequences: List[List[str]] = []
    metadata_records: List[Dict[str, Any]] = []
    missing_tactics: Dict[str, int] = {}
    tactic_mapping = tactic_mapping or {}

    if not tactic_mapping:
        raise RuntimeError(
            "Tactic mapping is required to build tactic-aware sequences but none was provided."
        )

    # Track drop reasons
    drop_reasons = {
        'sequence_too_short': 0,
        'no_tactic_coverage': 0,
    }

    # Check if this is the v2 format with pre-built sequences
    if 'sequence' in df.columns:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
            sequence = row['sequence']
            original_length = len(sequence)
            
            if original_length < 2:
                drop_reasons['sequence_too_short'] += 1
                continue

            tactic_sequence = _resolve_tactic_sequence(
                sequence, 
                tactic_mapping, 
                missing_tactics,
                skip_unmapped=skip_unmapped_techniques
            )
            if tactic_sequence is None:
                drop_reasons['no_tactic_coverage'] += 1
                continue

            sequence_id = len(sequences)
            sequences.append(sequence)

            record = {
                'sequence_id': sequence_id,
                'sequence_length': len(sequence),
                'unique_techniques': len(set(sequence)),
                'tactic_sequence': tactic_sequence,
                'kill_chain_boundaries': _derive_kill_chain_boundaries(tactic_sequence),
                'original_length': original_length,
                'filtered_length': len(sequence),
                'unmapped_count': original_length - len(tactic_sequence) if skip_unmapped_techniques else 0,
            }

            for col, value in row.items():
                if col in {'sequence'} or col in record:
                    continue
                record[col] = value

            metadata_records.append(record)
    else:
        # Process individual rows as sequences (no grouping)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
            if 'technique_id' not in row:
                continue
                
            sequence = [row['technique_id']]
            if len(sequence) < 2:
                drop_reasons['sequence_too_short'] += 1
                continue

            tactic_sequence = _resolve_tactic_sequence(
                sequence, 
                tactic_mapping, 
                missing_tactics,
                skip_unmapped=skip_unmapped_techniques
            )
            if tactic_sequence is None:
                drop_reasons['no_tactic_coverage'] += 1
                continue

            sequence_id = len(sequences)
            sequences.append(sequence)

            metadata_records.append({
                'sequence_id': sequence_id,
                'sequence_length': len(sequence),
                'unique_techniques': len(set(sequence)),
                'tactic_sequence': tactic_sequence,
                'kill_chain_boundaries': _derive_kill_chain_boundaries(tactic_sequence),
                'original_length': len(sequence),
                'filtered_length': len(sequence),
                'unmapped_count': 0,
            })

    metadata_df = pd.DataFrame(metadata_records)

    if not metadata_df.empty:
        print(f"\n✅ Built {len(sequences)} sequences")
        print(f"   Average sequence length: {metadata_df['sequence_length'].mean():.2f}")
        print(f"   Average unique techniques: {metadata_df['unique_techniques'].mean():.2f}")
        print(f"   Min sequence length: {metadata_df['sequence_length'].min()}")
        print(f"   Max sequence length: {metadata_df['sequence_length'].max()}")
        
        if skip_unmapped_techniques:
            total_unmapped = metadata_df['unmapped_count'].sum()
            print(f"   Total unmapped techniques skipped: {total_unmapped}")
            if len(metadata_df) > 0:
                print(f"   Average unmapped per sequence: {metadata_df['unmapped_count'].mean():.2f}")

    # Print drop diagnostics
    print(f"\n📊 Sequence Drop Analysis:")
    total_input = len(df)
    for reason, count in drop_reasons.items():
        if count > 0:
            pct = 100 * count / total_input if total_input > 0 else 0
            print(f"   ❌ {reason}: {count} ({pct:.1f}%)")

    # Print unmapped techniques summary
    if missing_tactics:
        print(f"\n⚠️  Top 10 Unmapped Techniques:")
        for tid, count in sorted(missing_tactics.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(df) if len(df) > 0 else 0
            print(f"   - {tid}: {count} occurrences ({pct:.1f}%)")

    return sequences, metadata_df


def _log_windows_sequence_breakdown(
    df: pd.DataFrame,
    combined_filename: str = WINDOWS_APT_COMBINED_FILENAME,
) -> None:
    """Summarize how many Windows sequences came from combined.csv vs. daily dumps."""

    total_sequences = len(df)
    combined_count = 0
    if 'source_path' in df.columns:
        combined_mask = (
            df['source_path']
            .fillna('')
            .apply(lambda value: Path(str(value)).name.lower())
            == combined_filename.lower()
        )
        combined_count = int(combined_mask.sum())

    daily_count = total_sequences - combined_count
    print(
        "Windows APT telemetry contribution: "
        f"{total_sequences} sequences total | "
        f"{combined_count} from {combined_filename} (_source.rule.mitre.id) | "
        f"{daily_count} from per-day exports"
    )

    if combined_count == 0:
        print(
            "WARNING: combined.csv did not contribute any sequences. "
            "Ensure the _source.rule.mitre.id column is populated and the Git LFS data "
            "has been downloaded."
        )


def _refresh_tactic_transition_artifact(sequences: List[List[str]]) -> None:
    """Recompute tactic transitions while shielding preprocessing from failures."""

    if not sequences:
        print("Skipping tactic transition update because no sequences were produced.")
        return

    try:
        update_tactic_transitions_from_sequences(
            sequences,
            mapping_path=cfg.TACTIC_MAPPING_JSON,
            output_path=cfg.TACTIC_TRANSITIONS_JSON,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(
            "WARNING: Failed to refresh tactic transition matrix. "
            f"Reason: {exc}"
        )


_PRUNED_COLUMNS = [
    'unique_techniques',
    'source',
    'source_path',
    'source_dataset',
]


def _finalize_metadata_table(
    sequences: List[List[str]],
    metadata: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Attach bookkeeping columns."""

    if metadata is None or metadata.empty:
        metadata = pd.DataFrame(
            columns=[
                'sequence_id',
                'sequence_length',
                'unique_techniques',
            ]
        )
    else:
        metadata = metadata.copy()

    metadata['source'] = source
    if 'sequence_id' in metadata.columns:
        metadata['source_sequence_id'] = metadata['sequence_id']
    else:
        metadata['source_sequence_id'] = np.arange(len(metadata))
        metadata['sequence_id'] = metadata['source_sequence_id']

    metadata['sequence'] = [
        [str(tech) for tech in seq]
        for seq in sequences
    ]

    metadata.reset_index(drop=True, inplace=True)
    metadata['sequence_id'] = metadata.index

    return metadata


def _prune_sequence_metadata_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Drop source bookkeeping columns from the exported parquet.

    Users requested to remove data-source breadcrumbs from
    the saved sequence metadata while leaving the in-memory view unchanged for
    sorting and diagnostics.
    """

    missing = [col for col in _PRUNED_COLUMNS if col not in table.columns]
    if len(missing) == len(_PRUNED_COLUMNS):
        return table

    return table.drop(columns=[col for col in _PRUNED_COLUMNS if col in table.columns])


def compute_ngram_counts(sequences: List[List[str]], n: int) -> pd.DataFrame:
    """
    Compute n-gram counts from sequences.
    
    Args:
        sequences: List of technique sequences
        n: N-gram size (1 for unigrams, 2 for bigrams, 3 for trigrams)
        
    Returns:
        DataFrame with n-gram counts
    """
    print(f"Computing {n}-gram counts...")
    
    ngram_counts = defaultdict(int)
    
    for sequence in tqdm(sequences, desc=f"Processing {n}-grams"):
        if len(sequence) < n:
            continue
        
        # Generate n-grams
        for i in range(len(sequence) - n + 1):
            if n == 1:
                ngram = sequence[i]
                context = []
            elif n == 2:
                context = [sequence[i]]
                ngram = sequence[i + 1]
            elif n == 3:
                context = [sequence[i], sequence[i + 1]]
                ngram = sequence[i + 2]
            else:
                # For higher-order n-grams
                context = sequence[i:i + n - 1]
                ngram = sequence[i + n - 1]
            
            # Create context key
            context_key = tuple(context) if context else ()
            ngram_counts[(context_key, ngram)] += 1
    
    # Convert to DataFrame
    data = []
    for (context, ngram), count in ngram_counts.items():
        data.append({
            'context': list(context),
            'candidate': ngram,
            'count': count
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('count', ascending=False)
    
    print(f"Computed {len(df)} unique {n}-grams")
    print(f"Total {n}-gram occurrences: {df['count'].sum()}")
    
    return df


def preprocess_sightings(
    legacy_csv: Path,
    combined_output: Path,
    data_source: Optional[str] = None,
    windows_dataset_dir: Optional[Path] = None,
    per_source_outputs: Optional[Dict[str, Path]] = None,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """Preprocess one or more telemetry corpora into canonical sequences."""

    selection = normalize_data_source(data_source)
    sources = list(iter_data_sources(selection))
    outputs = per_source_outputs or {
        'legacy': cfg.SEQUENCE_OUTPUT_MAP['legacy'],
        'windows-apt': cfg.SEQUENCE_OUTPUT_MAP['windows-apt'],
    }

    ensure_dir(combined_output.parent)
    for path in outputs.values():
        ensure_dir(Path(path).parent)

    tactic_mapping = load_tactic_mapping(cfg.TACTIC_MAPPING_JSON)
    if not tactic_mapping:
        raise RuntimeError(
            "Tactic mapping is required but could not be loaded. "
            f"Expected file at {cfg.TACTIC_MAPPING_JSON}."
        )

    processed_tables: Dict[str, pd.DataFrame] = {}

    if 'legacy' in sources:
        print("Loading legacy sightings dataset...")
        legacy_df = load_sightings_data(legacy_csv)
        print(f"Loaded {len(legacy_df)} legacy rows")
        legacy_mapping = build_technique_mapping(legacy_df)
        legacy_df_mapped = apply_technique_mapping(legacy_df, legacy_mapping)
        legacy_sequences, legacy_meta = build_sequences(legacy_df_mapped, tactic_mapping)
        legacy_table = _finalize_metadata_table(legacy_sequences, legacy_meta, 'legacy')
        processed_tables['legacy'] = legacy_table

    if 'windows-apt' in sources:
        dataset_dir = windows_dataset_dir or cfg.WINDOWS_APT_DATASET_DIR
        if dataset_dir is None or not Path(dataset_dir).exists():
            raise FileNotFoundError(
                "Windows APT dataset directory not found. Configure WINDOWS_APT_DATASET_DIR or "
                "provide windows_dataset_dir explicitly."
            )

        print("Loading Windows APT telemetry...")
        windows_df = parse_windows_apt_sequences(dataset_dir)
        print(f"Loaded {len(windows_df)} Windows APT sequences")
        _log_windows_sequence_breakdown(windows_df)
        windows_mapping = build_technique_mapping(windows_df)
        windows_df_mapped = apply_technique_mapping(windows_df, windows_mapping)
        windows_sequences, windows_meta = build_sequences(windows_df_mapped, tactic_mapping)
        windows_table = _finalize_metadata_table(windows_sequences, windows_meta, 'windows-apt')
        processed_tables['windows-apt'] = windows_table

    if not processed_tables:
        raise RuntimeError("No data sources were processed during preprocessing.")

    missing_sources = [source for source in sources if source not in processed_tables]
    if missing_sources:
        raise RuntimeError(
            "The following data sources were requested but produced no sequences: "
            + ", ".join(sorted(missing_sources))
        )

    combined_table = pd.concat(processed_tables.values(), ignore_index=True)
    combined_table.reset_index(drop=True, inplace=True)
    combined_table['sequence_id'] = combined_table.index

    pruned_tables = {
        name: _prune_sequence_metadata_columns(table)
        for name, table in processed_tables.items()
    }
    pruned_combined = _prune_sequence_metadata_columns(combined_table)

    for source_name, table in pruned_tables.items():
        if source_name in outputs:
            save_parquet(table, outputs[source_name])
            print(f"Saved {source_name} sequences to {outputs[source_name]}")

    save_parquet(pruned_combined, combined_output)
    print(f"Saved combined sequences to {combined_output}")

    _refresh_tactic_transition_artifact(combined_table['sequence'].tolist())

    summary_parts = [
        f"{source}={len(table)}"
        for source, table in processed_tables.items()
    ]
    print(
        "Per-source sequence counts (post-merging): "
        + ", ".join(summary_parts)
        + f" | total={len(combined_table)}"
    )

    selected_table = pruned_combined if selection == 'both' else pruned_tables[selection]
    sequences_out = [list(map(str, seq)) for seq in selected_table['sequence']]
    metadata_out = selected_table.drop(columns=['sequence']).copy()

    return sequences_out, metadata_out


def main():
    """Main function to preprocess sightings data."""
    # Ensure output directory exists
    ensure_dir(cfg.SEQUENCES_PARQUET.parent)

    # Preprocess sightings
    sequences, metadata = preprocess_sightings(
        cfg.SIGHTINGS_CSV,
        cfg.SEQUENCES_PARQUET,
        data_source=cfg.DEFAULT_DATA_SOURCE,
    )

    print(
        "Preprocessing complete. Generated "
        f"{len(sequences)} sequences."
    )


if __name__ == "__main__":
    main()
