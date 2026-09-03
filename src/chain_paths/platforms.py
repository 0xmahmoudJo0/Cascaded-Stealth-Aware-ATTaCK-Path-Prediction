"""Platform mapping utilities for ATT&CK techniques.

Extracts technique->platform lists from the enterprise ATT&CK STIX bundle
so prediction can constrain outputs to Windows/AD when desired.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from . import config as cfg


def _normalize_tid(value: str) -> str:
    return str(value).strip().upper()


def _normalize_platform(value: str) -> str:
    return str(value).strip().lower()


def _extract_technique_id(obj: Mapping[str, object]) -> str | None:
    """Pull the MITRE external_id (e.g., T1059.001) from a STIX object."""
    for ref in obj.get("external_references", []) or []:
        if not isinstance(ref, Mapping):
            continue
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return _normalize_tid(str(ref["external_id"]))
    return None


def _extract_platforms(obj: Mapping[str, object]) -> List[str]:
    platforms: List[str] = []
    raw_platforms: Iterable[str] = obj.get("x_mitre_platforms") or obj.get("x_mitre_platforms", []) or []
    for platform in raw_platforms:
        if not platform:
            continue
        platforms.append(_normalize_platform(str(platform)))
    return platforms


def load_platform_mapping(path: Path | None = None) -> Dict[str, List[str]]:
    """Load technique -> platform mapping from enterprise-attack.json.

    Gracefully returns an empty mapping when the file is missing or malformed.
    """
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

        tid = _extract_technique_id(obj)
        if not tid:
            continue

        platforms = _extract_platforms(obj)
        if not platforms:
            continue

        mapping[tid] = sorted(set(platforms))

    return mapping


__all__ = ["load_platform_mapping"]
