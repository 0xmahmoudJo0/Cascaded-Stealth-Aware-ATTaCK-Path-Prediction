"""Utilities for tactic-aware smoothing and transition modeling."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import config as cfg
from .io import ensure_dir

START_TOKEN = "__start__"


def _normalize_tid(value: str) -> str:
    return str(value).strip().upper()


def _normalize_tactic(value: str) -> str:
    return str(value).strip().lower()


def _extract_technique_id(obj: Mapping[str, object]) -> Optional[str]:
    """Pull the MITRE external_id (e.g., T1059.001) from a STIX object."""
    for ref in obj.get("external_references", []) or []:
        if not isinstance(ref, Mapping):
            continue
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return _normalize_tid(str(ref["external_id"]))
    return None


def _extract_tactics(obj: Mapping[str, object]) -> List[str]:
    tactics: List[str] = []
    for phase in obj.get("kill_chain_phases", []) or []:
        if not isinstance(phase, Mapping):
            continue
        if phase.get("kill_chain_name") != "mitre-attack":
            continue
        phase_name = phase.get("phase_name")
        if not phase_name:
            continue
        tactics.append(_normalize_tactic(str(phase_name)))
    return tactics


@lru_cache(maxsize=1)
def _load_mitre_tactic_mapping(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load technique -> tactic mapping from enterprise-attack.json."""
    bundle_path = Path(path or cfg.ENTERPRISE_ATTACK_JSON)
    if not bundle_path.exists():
        return {}

    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    objects: Sequence[Mapping[str, object]]
    if isinstance(payload, Mapping):
        objects = payload.get("objects", []) or []
    elif isinstance(payload, Sequence):
        objects = payload
    else:
        return {}

    mapping: Dict[str, List[str]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        if obj.get("type") not in {"attack-pattern", "x-mitre-attack-pattern"}:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        tid = _extract_technique_id(obj)
        if not tid:
            continue

        tactics = _extract_tactics(obj)
        if not tactics:
            continue

        mapping[tid] = sorted(set(tactics))

    return mapping


def load_tactic_mapping(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load the technique-to-tactic lookup if it exists on disk."""

    mapping_path = Path(path or cfg.TACTIC_MAPPING_JSON)
    normalized: Dict[str, List[str]] = {}
    if mapping_path.exists():
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Failed to parse tactic mapping at {mapping_path}: {exc}") from exc

        for key, values in payload.items():
            tid = _normalize_tid(key)
            if not values:
                continue
            tactics = sorted({
                _normalize_tactic(tactic)
                for tactic in (values if isinstance(values, (list, tuple)) else [values])
                if str(tactic).strip()
            })
            if tactics:
                normalized[tid] = tactics

    stix_mapping = _load_mitre_tactic_mapping()
    if not stix_mapping:
        return normalized
    if not normalized:
        return stix_mapping

    merged: Dict[str, List[str]] = {}
    for tid, tactics in stix_mapping.items():
        merged[tid] = list(tactics)
    for tid, tactics in normalized.items():
        if tid in merged:
            merged[tid] = sorted(set(merged[tid]) | set(tactics))
        else:
            merged[tid] = tactics

    return merged


def load_tactic_transitions(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Load the persisted tactic transition probabilities."""

    transition_path = Path(path or cfg.TACTIC_TRANSITIONS_JSON)
    if not transition_path.exists():
        return {}

    try:
        payload = json.loads(transition_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Failed to parse tactic transitions at {transition_path}: {exc}") from exc

    if isinstance(payload, dict) and 'transitions' in payload:
        transitions = payload.get('transitions', {})
    else:
        transitions = payload

    normalized: Dict[str, Dict[str, float]] = {}
    for prev_tactic, destinations in transitions.items():
        if not isinstance(destinations, Mapping):
            continue
        normalized_prev = _normalize_tactic(prev_tactic)
        normalized[normalized_prev] = {
            _normalize_tactic(next_tactic): float(prob)
            for next_tactic, prob in destinations.items()
            if prob is not None
        }

    return normalized


def save_tactic_transitions(
    transitions: Mapping[str, Mapping[str, float]],
    path: Optional[Path] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Persist tactic transition probabilities as JSON."""

    output_path = Path(path or cfg.TACTIC_TRANSITIONS_JSON)
    ensure_dir(output_path.parent)

    payload = {
        'metadata': dict(metadata or {}),
        'transitions': transitions,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def compute_tactic_transition_counts(
    sequences: Iterable[Sequence[str]],
    mapping: Mapping[str, Sequence[str]],
    start_token: str = START_TOKEN,
    tactic_priors: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Aggregate tactic transition counts from technique sequences."""

    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    observed_tactics = set()
    start_key = _normalize_tactic(start_token)

    def build_weights(tactics: Sequence[str]) -> Dict[str, float]:
        normalized = [_normalize_tactic(tactic) for tactic in tactics]
        if not normalized:
            return {}
        if not tactic_priors:
            weight = 1.0 / len(normalized)
            return {tactic: weight for tactic in normalized}
        raw_weights: List[float] = []
        for tactic in normalized:
            weight = float(tactic_priors.get(tactic, 0.0))
            if weight <= 0:
                weight = 1e-6
            raw_weights.append(weight)
        total = sum(raw_weights)
        if total <= 0:
            weight = 1.0 / len(normalized)
            return {tactic: weight for tactic in normalized}
        return {
            tactic: raw_weight / total
            for tactic, raw_weight in zip(normalized, raw_weights)
        }

    for sequence in sequences:
        if not sequence:
            continue
        previous_tactics: Optional[Dict[str, float]] = None
        for technique in sequence:
            tid = _normalize_tid(technique)
            tactics = mapping.get(tid)
            if not tactics:
                previous_tactics = None
                continue

            weights = build_weights(tactics)
            if not weights:
                previous_tactics = None
                continue
            observed_tactics.update(weights.keys())

            if previous_tactics is None:
                for tactic, weight in weights.items():
                    counts[start_key][tactic] += weight
            else:
                for prev, prev_weight in previous_tactics.items():
                    for tactic, weight in weights.items():
                        counts[prev][tactic] += prev_weight * weight

            previous_tactics = weights

    return counts, len(observed_tactics)


def compute_tactic_priors(
    sequences: Iterable[Sequence[str]],
    mapping: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    """Estimate tactic priors from sequence usage to weight multi-tactic techniques."""
    counts: Dict[str, float] = defaultdict(float)

    for sequence in sequences:
        if not sequence:
            continue
        for technique in sequence:
            tid = _normalize_tid(technique)
            tactics = mapping.get(tid)
            if not tactics:
                continue
            normalized = [_normalize_tactic(tactic) for tactic in tactics]
            if not normalized:
                continue
            share = 1.0 / len(normalized)
            for tactic in normalized:
                counts[tactic] += share

    if not counts:
        for tactics in mapping.values():
            normalized = [_normalize_tactic(tactic) for tactic in tactics]
            if not normalized:
                continue
            share = 1.0 / len(normalized)
            for tactic in normalized:
                counts[tactic] += share

    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {tactic: value / total for tactic, value in counts.items()}


def normalize_tactic_transition_counts(
    counts: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Normalize transition counts into probabilities."""

    matrix: Dict[str, Dict[str, float]] = {}
    for prev, destinations in counts.items():
        total = float(sum(destinations.values()))
        if total <= 0:
            continue
        matrix[prev] = {
            tactic: value / total
            for tactic, value in destinations.items()
            if value > 0
        }
    return matrix


def update_tactic_transitions_from_sequences(
    sequences: Iterable[Sequence[str]],
    mapping_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    """Recompute and persist tactic transitions for the provided sequences."""

    sequence_list = list(sequences)
    if not sequence_list:
        print("Skipping tactic transition export because no sequences were provided.")
        return 0

    mapping = load_tactic_mapping(mapping_path)
    if not mapping:
        print(
            "Skipping tactic transition export because no mapping file was found. "
            f"Expected path: {mapping_path or cfg.TACTIC_MAPPING_JSON}"
        )
        return 0

    tactic_priors = compute_tactic_priors(sequence_list, mapping)
    counts, tactic_count = compute_tactic_transition_counts(
        sequence_list,
        mapping,
        tactic_priors=tactic_priors,
    )
    if not counts:
        print(
            "Tactic mapping loaded but no overlapping transitions were derived from the sequences."
        )
        return 0

    transitions = normalize_tactic_transition_counts(counts)
    edge_count = sum(len(dest) for dest in transitions.values())

    metadata = {
        'sequence_count': len(sequence_list),
        'tactic_count': tactic_count,
        'edge_count': edge_count,
        'start_token': START_TOKEN,
    }

    destination = save_tactic_transitions(transitions, path=output_path, metadata=metadata)
    print(
        "Saved tactic transition matrix "
        f"with {edge_count} edges spanning {tactic_count} tactics to "
        f"{destination}."
    )
    return edge_count
