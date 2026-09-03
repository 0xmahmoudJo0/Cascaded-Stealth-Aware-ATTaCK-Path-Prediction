"""File I/O utilities for the chain-aware predictor pipeline."""

import json
import re
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:  # pragma: no cover - import guard for standalone utilities
    from . import config as cfg
except ImportError:  # pragma: no cover - allows running module as a script
    cfg = None  # type: ignore

from .technique_utils import canonicalize_technique_id


WINDOWS_APT_COMBINED_FILENAME = (
    getattr(cfg, "WINDOWS_APT_COMBINED_FILENAME", "combined.csv")
    if cfg is not None
    else "combined.csv"
)
_COMBINED_SEQUENCE_COLUMN = "_source.rule.mitre.id"
_GIT_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


_COMBINED_SEQUENCE_COLUMN = "_source.rule.mitre.id"

# Use string literal for type hint to avoid circular import
if sys.version_info >= (3, 7):
    from typing import TYPE_CHECKING


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        return pd.read_csv(filepath, low_memory=False, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to load CSV {filepath}: {e}")


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV with directory creation."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_csv(filepath, index=False, **kwargs)


def load_parquet(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load Parquet file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    try:
        return pd.read_parquet(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to load Parquet {filepath}: {e}")


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to Parquet with directory creation."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_parquet(filepath, index=False, **kwargs)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load JSON {filepath}: {e}")


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with directory creation."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load NumPy array with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NumPy file not found: {filepath}")
    
    try:
        return np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to load NumPy array {filepath}: {e}")


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save NumPy array with directory creation."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    np.save(filepath, array)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load pickle {filepath}: {e}")


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file with directory creation."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(filepath: Union[str, Path]) -> Any:
    """Load YAML file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Failed to load YAML {filepath}: {e}")


def load_yaml_all(filepath: Union[str, Path]) -> List[Any]:
    """Load all YAML documents from file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(yaml.safe_load_all(f))
    except Exception as e:
        raise IOError(f"Failed to load YAML {filepath}: {e}")


def find_yaml_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find all YAML files in directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pattern = "**/*.yml" if recursive else "*.yml"
    yml_files = list(directory.glob(pattern))
    
    pattern = "**/*.yaml" if recursive else "*.yaml"
    yaml_files = list(directory.glob(pattern))
    
    return sorted(yml_files + yaml_files)


def load_sightings_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load and validate sightings data."""
    df = load_csv(filepath)
    
    # Check if this is the v2 format with tid and sighting_date columns
    if 'tid' in df.columns and 'sighting_date' in df.columns:
        # This is the v2 format - process it into the standard sequence format.
        # Using vectorized operations is much faster than iterrows().

        # Drop rows with malformed 'tid' strings before processing.
        df = df[df['tid'].str.match(r"^\[.*\]$", na=False)].copy()

        # Convert string representation of list to an actual list of strings.
        # This is faster and safer than ast.literal_eval for this specific format.
        df['sequence'] = df['tid'].str.strip('[]').str.replace("'", "").str.split(', ')

        # Filter for sequences with at least 2 techniques.
        df = df[df['sequence'].str.len() >= 2].copy()

        # Generate new columns.
        df['sighting_id'] = [f"s_{i}" for i in df.index]

        # Select and reorder columns to match the standard format
        df = df[['sighting_id', 'sequence']].copy()
        return df
    
    return df


# ---------------------------------------------------------------------------
# Windows APT 2025 dataset helpers
# ---------------------------------------------------------------------------


def _normalize_column_name(name: Any) -> str:
    """Normalize heterogeneous column names for resilient lookups."""

    text = "" if name is None else str(name)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _resolve_column(
    candidates: Iterable[str],
    available: Dict[str, str],
) -> Optional[str]:
    """Resolve the first matching normalized column name from a list of candidates."""

    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def _collect_manifest_techniques(row: pd.Series, technique_columns: List[str]) -> List[str]:
    """Extract unique technique strings from manifest rows."""

    values: List[str] = []
    for col in technique_columns:
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, str):
            # Split on commas, semicolons, pipes, and whitespace combinations.
            parts = re.split(r"[;,\n\r\t|]+", val)
            values.extend(part.strip() for part in parts if part and part.strip())
        elif isinstance(val, (list, tuple, set)):
            values.extend(str(item).strip() for item in val if str(item).strip())
        else:
            text = str(val).strip()
            if text:
                values.append(text)

    # Deduplicate while preserving order.
    seen = set()
    unique_values: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def enumerate_windows_apt_scenarios(
    manifest_path: Union[str, Path]
) -> pd.DataFrame:
    """Parse the Windows APT 2025 scenario manifest and enumerate metadata.

    Parameters
    ----------
    manifest_path:
        Path to ``scenario_manifest.csv`` from the Windows APT 2025 dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns:

        ``scenario_id``
            Scenario identifier supplied by the dataset.

        ``group_id``
            Best-effort adversary/group label associated with the scenario.

        ``scenario_name``
            Human-readable description/title of the scenario, when available.

        ``raw_techniques``
            All ATT&CK technique tags provided by the manifest.

        ``canonical_techniques``
            Canonicalized ATT&CK technique IDs produced via
            :func:`~src.chain_paths.preprocess.canonicalize_technique_id`.
    """

    manifest_path = Path(manifest_path)
    manifest_df = load_csv(manifest_path)
    normalized_lookup = {
        _normalize_column_name(col): col for col in manifest_df.columns
    }

    scenario_column = _resolve_column(
        (
            "scenario_id",
            "scenario",
            "scenario_identifier",
            "scenario_number",
            "scenario_name",
        ),
        normalized_lookup,
    )
    if not scenario_column:
        raise ValueError(
            "Unable to locate a scenario identifier column within the manifest."
        )

    group_column = _resolve_column(
        (
            "adversary",
            "threat_group",
            "group",
            "apt_group",
            "adversary_group",
            "threat_actor",
        ),
        normalized_lookup,
    )

    name_column = _resolve_column(
        (
            "scenario_name",
            "name",
            "title",
            "scenario_description",
            "description",
        ),
        normalized_lookup,
    )

    phase_column = _resolve_column(
        (
            "scenario_phase",
            "phase",
            "exercise_phase",
            "campaign_phase",
            "operation_phase",
        ),
        normalized_lookup,
    )

    persistence_column = _resolve_column(
        (
            "persistence_focus",
            "persistence",
            "persistence_strategy",
            "persistence_goal",
        ),
        normalized_lookup,
    )

    # Gather all technique related columns.
    technique_columns = [
        normalized_lookup[col]
        for col in normalized_lookup
        if "technique" in col or col in {"attack_id", "mitre_attack_id", "tid"}
    ]

    if not technique_columns:
        technique_columns = []

    records: List[Dict[str, Any]] = []
    for _, row in manifest_df.iterrows():
        scenario_id_raw = row.get(scenario_column)
        scenario_id = str(scenario_id_raw).strip() if not pd.isna(scenario_id_raw) else ""
        if not scenario_id:
            continue

        scenario_name = row.get(name_column) if name_column else None
        scenario_name = (
            str(scenario_name).strip()
            if scenario_name is not None and not pd.isna(scenario_name)
            else None
        )

        scenario_phase = None
        if phase_column:
            phase_value = row.get(phase_column)
            if phase_value is not None and not pd.isna(phase_value):
                scenario_phase = str(phase_value).strip()

        persistence_focus = None
        if persistence_column:
            persistence_value = row.get(persistence_column)
            if persistence_value is not None and not pd.isna(persistence_value):
                persistence_focus = str(persistence_value).strip()

        raw_techniques = _collect_manifest_techniques(row, technique_columns)
        canonical_techniques: List[str] = []
        for tech in raw_techniques:
            canonical = canonicalize_technique_id(tech)
            if canonical:
                canonical_techniques.append(canonical)

        records.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "scenario_phase": scenario_phase,
                "persistence_focus": persistence_focus,
                "raw_techniques": raw_techniques,
                "canonical_techniques": canonical_techniques,
            }
        )

    if not records:
        raise ValueError("Scenario manifest did not yield any scenario entries.")

    manifest_summary = pd.DataFrame.from_records(records)
    manifest_summary.sort_values("scenario_id", inplace=True)
    manifest_summary.reset_index(drop=True, inplace=True)
    return manifest_summary


def _prepare_windows_apt_event_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalize Windows APT event columns and return lookup mapping."""

    normalized_lookup = {_normalize_column_name(col): col for col in df.columns}
    return df, normalized_lookup


def _extract_scenario_series(
    df: pd.DataFrame,
    lookup: Dict[str, str],
) -> Tuple[pd.Series, Optional[str]]:
    """Retrieve scenario identifiers from an event frame."""

    scenario_column = _resolve_column(
        (
            "scenario_id",
            "scenario",
            "scenario_identifier",
            "campaign_id",
            "campaign",
            "exercise",
        ),
        lookup,
    )

    if scenario_column is None:
        return pd.Series([None] * len(df)), None

    series = df[scenario_column].astype(str).str.strip()
    series = series.replace({"": None})
    return series, scenario_column


def _extract_technique_column_candidates(lookup: Dict[str, str]) -> List[str]:
    """Identify potential technique columns from a normalized lookup."""

    technique_columns: List[str] = []
    for normalized, original in lookup.items():
        if normalized in {"technique_id", "technique", "mitre_attack_id", "attack_id", "tid"}:
            technique_columns.append(original)
        elif "technique" in normalized or "attack_id" in normalized:
            technique_columns.append(original)
        elif normalized.endswith("_tid"):
            technique_columns.append(original)
    return technique_columns


def _parse_combined_sequence_cell(
    value: Any, canonicalize_fn: Any
) -> List[str]:
    """Extract ordered MITRE techniques from a combined.csv cell."""

    if pd.isna(value):
        return []

    if isinstance(value, list):
        candidates = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        candidates = re.findall(r"T\d{4}(?:\.\d+)?", text.upper())
        if not candidates and "," in text:
            candidates = [part.strip() for part in text.split(",") if part.strip()]

    sequence: List[str] = []
    for candidate in candidates:
        canonical = canonicalize_fn(candidate)
        if canonical:
            sequence.append(canonical)

    return sequence


def _parse_combined_sequence_file(
    csv_path: Path,
    canonicalize_fn: Any,
    source_dataset: str,
    dataset_dir: Path,
) -> List[Dict[str, Any]]:
    """Parse combined.csv exports that already contain technique sequences."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")

    print(
        "Reading combined telemetry export and extracting technique sequences "
        f"from '{_COMBINED_SEQUENCE_COLUMN}'..."
    )

    records: List[Dict[str, Any]] = []
    chunk_iter = pd.read_csv(
        csv_path,
        usecols=[_COMBINED_SEQUENCE_COLUMN],
        chunksize=50_000,
        dtype=str,
        low_memory=False,
        encoding_errors='ignore',
    )

    for chunk in chunk_iter:
        sequences = chunk[_COMBINED_SEQUENCE_COLUMN].apply(
            lambda value: _parse_combined_sequence_cell(value, canonicalize_fn)
        )

        for sequence in sequences:
            if len(sequence) < 2:
                continue
            sequence_id = len(records)
            records.append(
                {
                    "sequence": sequence,
                    "source_dataset": source_dataset,
                    "source_path": str(csv_path.relative_to(dataset_dir)),
                }
            )

    print(f"Extracted {len(records)} sequences from {csv_path.name}.")
    return records


def parse_windows_apt_sequences(
    dataset_dir: Union[str, Path],
    combined_filename: str = WINDOWS_APT_COMBINED_FILENAME,
    source_dataset: str = "windows_apt_2025",
) -> pd.DataFrame:
    """Parse Windows APT 2025 telemetry into canonical sequences.

    Parameters
    ----------
    dataset_dir:
        Path to the directory containing the Windows APT 2025 CSV exports.

    combined_filename:
        Optional name of the combined CSV (if synced via Git LFS). When the
        combined file is present it will be used instead of individual daily
        exports.

    source_dataset:
        Provenance identifier added to the resulting frame for downstream
        filtering.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``sequence`` and provenance metadata such as
        ``source_dataset`` and ``source_path``.
    """

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    manifest_path = dataset_dir / "scenario_manifest.csv"
    manifest_summary = None
    manifest_lookup: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        manifest_summary = enumerate_windows_apt_scenarios(manifest_path)
        manifest_lookup = manifest_summary.set_index("scenario_id").to_dict("index")

    combined_path = dataset_dir / combined_filename
    remaining_csvs: List[Path] = [
        csv_path
        for csv_path in sorted(dataset_dir.glob("*.csv"))
        if csv_path.name.lower() not in {"scenario_manifest.csv", "validation_summary.csv"}
        and csv_path.name != combined_filename
    ]

    if not combined_path.exists() and not remaining_csvs:
        raise FileNotFoundError(
            "No Windows APT 2025 CSV files found. Ensure Git LFS data is synced."
        )

    all_records: List[Dict[str, Any]] = []
    if combined_path.exists():
        combined_records = _parse_combined_sequence_file(
            combined_path,
            canonicalize_technique_id,
            source_dataset,
            dataset_dir,
        )
        all_records.extend(combined_records)

    for csv_path in remaining_csvs:
        df = load_csv(csv_path)
        if df.empty:
            continue

        df, lookup = _prepare_windows_apt_event_frame(df)
        scenario_series, scenario_column = _extract_scenario_series(df, lookup)
        technique_columns = _extract_technique_column_candidates(lookup)

        if not technique_columns:
            raise ValueError(
                f"Unable to locate technique columns in {csv_path.name}."
            )

        # Build canonical technique IDs per row.
        canonical_tids: List[Optional[str]] = []
        for _, row in df.iterrows():
            canonical_id: Optional[str] = None
            for column in technique_columns:
                value = row.get(column)
                if pd.isna(value):
                    continue
                canonical_id = canonicalize_technique_id(str(value))
                if canonical_id:
                    break
            canonical_tids.append(canonical_id)

        df = df.assign(
            _scenario=scenario_series,
            _canonical_tid=canonical_tids,
            _row_order=np.arange(len(df)),
        )

        # Determine fallback scenario identifier from filename if necessary.
        if scenario_column is None:
            df["_scenario"] = df["_scenario"].fillna(Path(csv_path).stem)

        # Group by scenario identifier (campaign surrogate).
        for scenario_id, scenario_df in df.groupby("_scenario"):
            if scenario_id is None or scenario_df.empty:
                continue

            ordered = scenario_df.sort_values(
                by=["_row_order"],
                kind="mergesort",
            )
            sequence = [tid for tid in ordered["_canonical_tid"].tolist() if tid]
            if not sequence:
                continue

            manifest_meta = manifest_lookup.get(str(scenario_id))
            scenario_name = None
            scenario_phase = None
            persistence_focus = None
            canonical_manifest_techniques: Optional[List[str]] = None
            if manifest_meta:
                scenario_name = manifest_meta.get("scenario_name")
                scenario_phase = manifest_meta.get("scenario_phase")
                persistence_focus = manifest_meta.get("persistence_focus")
                canonical_manifest_techniques = manifest_meta.get(
                    "canonical_techniques"
                )

            record: Dict[str, Any] = {
                "sequence": sequence,
                "source_dataset": source_dataset,
                "source_path": str(csv_path.relative_to(dataset_dir)),
            }

            if scenario_name:
                record["scenario_name"] = scenario_name
            if scenario_phase:
                record["scenario_phase"] = scenario_phase
            if persistence_focus:
                record["persistence_focus"] = persistence_focus
            if canonical_manifest_techniques is not None:
                record["manifest_canonical_techniques"] = canonical_manifest_techniques

            all_records.append(record)

    if not all_records:
        raise ValueError(
            "Windows APT CSV files did not yield any canonical technique sequences."
        )

    result_df = pd.DataFrame.from_records(all_records)
    return result_df


def load_mcdm_scores(filepath: Union[str, Path]) -> Dict[str, float]:
    """Load MCDM priority scores as dictionary."""
    df = load_csv(filepath)
    
    # Normalize column names to lowercase and replace spaces/special chars with underscores
    df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]+', '_', regex=True)
    
    # Check for common variations of 'technique_id' and 'priority_score'
    # and rename them to the standard 'technique_id' and 'priority_score'
    
    # For 'technique_id'
    if 'technique_id' not in df.columns:
        for col in ['technique_id', 'techniqueid', 'technique_id_', 'tid']:
            if col in df.columns:
                df.rename(columns={col: 'technique_id'}, inplace=True)
                break
    
    # For 'priority_score'
    if 'priority_score' not in df.columns:
        for col in ['priority_score', 'priorityscore', 'score', 'priority']:
            if col in df.columns:
                df.rename(columns={col: 'priority_score'}, inplace=True)
                break

    # Validate required columns
    if 'technique_id' not in df.columns or 'priority_score' not in df.columns:
        raise ValueError(
            f"MCDM scores file '{filepath}' must contain 'technique_id' and 'priority_score' columns. "
            f"Found columns: {list(df.columns)}"
        )
    
    return dict(zip(df['technique_id'], df['priority_score']))


def save_sequences(sequences: List[List[str]], filepath: Union[str, Path]) -> None:
    """Save sequences as Parquet file."""
    # Convert to DataFrame for easier handling
    data = []
    for i, seq in enumerate(sequences):
        data.append({
            'sequence_id': i,
            'sequence': seq,
            'length': len(seq)
        })
    
    df = pd.DataFrame(data)
    save_parquet(df, filepath)


def _normalize_sequence_column(raw_seq: Any) -> List[str]:
    if isinstance(raw_seq, list):
        seq_list = raw_seq
    elif isinstance(raw_seq, np.ndarray):
        seq_list = raw_seq.tolist()
    else:
        seq_list = list(raw_seq)

    return [str(tech) for tech in seq_list]


def load_sequences(filepath: Union[str, Path]) -> List[List[str]]:
    """Load sequences from Parquet file."""

    df = load_parquet(filepath)
    return [_normalize_sequence_column(raw_seq) for raw_seq in df['sequence']]


def load_tactic_sequences(filepath: Union[str, Path]) -> List[List[str]]:
    """Load tactic sequences from Parquet file if present."""

    df = load_parquet(filepath)
    if 'tactic_sequence' not in df.columns:
        return []

    return [_normalize_sequence_column(raw_seq) for raw_seq in df['tactic_sequence']]


def load_sequence_metadata(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load sequence-level metadata generated during preprocessing."""

    df = load_parquet(filepath)
    metadata_cols = [col for col in df.columns if col != 'sequence']
    metadata = df[metadata_cols].copy()

    return metadata


def resolve_sequences_path(data_source: str) -> Path:
    """Return the Parquet artifact corresponding to ``data_source``."""

    normalized = cfg.normalize_data_source(data_source)
    return cfg.SEQUENCE_OUTPUT_MAP[normalized]


def load_sequences_for_source(data_source: str) -> List[List[str]]:
    """Load technique sequences for the requested data source."""

    return load_sequences(resolve_sequences_path(data_source))


def load_sequence_metadata_for_source(data_source: str) -> pd.DataFrame:
    """Load sequence metadata for the requested data source."""

    return load_sequence_metadata(resolve_sequences_path(data_source))


def load_tactic_sequences_for_source(data_source: str) -> List[List[str]]:
    """Load tactic sequences for the requested data source if present."""

    return load_tactic_sequences(resolve_sequences_path(data_source))


def save_tech_index(tech_to_idx: Dict[str, int], filepath: Union[str, Path]) -> None:
    """Save technique to index mapping."""
    df = pd.DataFrame([
        {'technique_id': tech, 'index': idx}
        for tech, idx in tech_to_idx.items()
    ])
    save_csv(df, filepath)


def load_tech_index(filepath: Union[str, Path]) -> Dict[str, int]:
    """Load technique to index mapping."""
    df = load_csv(filepath)
    
    # Handle both 'technique_id' and 'tech_id' column names
    if 'technique_id' in df.columns:
        tech_col = 'technique_id'
    elif 'tech_id' in df.columns:
        tech_col = 'tech_id'
    else:
        raise ValueError(
            f"Tech index file '{filepath}' must contain 'technique_id' or 'tech_id' column. "
            f"Found columns: {list(df.columns)}"
        )
    
    if 'index' not in df.columns:
        raise ValueError(
            f"Tech index file '{filepath}' must contain 'index' column. "
            f"Found columns: {list(df.columns)}"
        )
    
    return dict(zip(df[tech_col], df['index']))


def save_paths_json(paths_data: List[Dict], filepath: Union[str, Path]) -> None:
    """Save Top-K paths in the required JSON format."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(paths_data, f, indent=2, ensure_ascii=False)


def load_paths_json(filepath: Union[str, Path]) -> List[Dict]:
    """Load Top-K paths from JSON file."""
    return load_json(filepath)
