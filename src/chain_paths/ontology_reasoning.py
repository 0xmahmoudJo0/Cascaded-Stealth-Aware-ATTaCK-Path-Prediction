"""Ontology-driven causal reasoning utilities.

This module ingests a lightweight ontology of tactics, techniques, and assets
and applies simplified SWRL-style rules to infer causal chains and parallel
branches within observed technique sequences. The resulting annotations can be
fed into downstream fusion/prediction components and persisted for auditing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _normalize_tid(value: str) -> str:
    return str(value).strip().upper()


@dataclass
class OntologyRule:
    """Simplified SWRL-style rule definition."""

    name: str
    antecedent: List[Mapping[str, Any]]
    consequent: Mapping[str, Any]
    weight: float = 1.0
    description: str = ""


@dataclass
class CausalAnnotation:
    source: str
    target: str
    relation: str = "causes"
    rule: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "rule": self.rule,
            "evidence": self.evidence,
        }


@dataclass
class EnhancedSequence:
    sequence: List[str]
    causal_annotations: List[CausalAnnotation]
    parallel_branches: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "causal_annotations": [ann.as_dict() for ann in self.causal_annotations],
            "parallel_branches": self.parallel_branches,
        }


class OntologyReasoner:
    """Applies ontology-backed causal reasoning to technique sequences."""

    def __init__(
        self,
        ontology: Mapping[str, Any],
        rules: Optional[Iterable[OntologyRule]] = None,
        *,
        default_relation: str = "causes",
    ) -> None:
        self.ontology: Dict[str, Any] = dict(ontology)
        self.default_relation = default_relation
        self.rules: List[OntologyRule] = list(rules or [])

        techniques = self.ontology.get("techniques", {})
        self._technique_metadata: Dict[str, Dict[str, Any]] = {
            _normalize_tid(k): (v or {}) for k, v in techniques.items()
        }
        self._relation_lookup: Dict[Tuple[str, str], str] = {}
        for relation in self.ontology.get("relations", []):
            src = relation.get("source")
            tgt = relation.get("target")
            rel = relation.get("relation", self.default_relation)
            if not src or not tgt:
                continue
            self._relation_lookup[(_normalize_tid(src), _normalize_tid(tgt))] = str(rel)

    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls, ontology_path: Path | str, rules_path: Optional[Path | str] = None
    ) -> "OntologyReasoner":
        with open(ontology_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        rules: List[OntologyRule] = []
        if rules_path and Path(rules_path).exists():
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                rules.append(
                    OntologyRule(
                        name=str(entry.get("name", "rule")),
                        antecedent=list(entry.get("antecedent", [])),
                        consequent=dict(entry.get("consequent", {})),
                        weight=float(entry.get("weight", 1.0)),
                        description=str(entry.get("description", "")),
                    )
                )
        return cls(ontology, rules)

    # ------------------------------------------------------------------
    def _conditions_hold(
        self, condition: Mapping[str, Any], technique: str, metadata: Mapping[str, Any]
    ) -> bool:
        field_name = condition.get("field", "technique")
        expected = condition.get("equals")
        if expected is None:
            return False

        if field_name == "technique":
            return _normalize_tid(technique) == _normalize_tid(expected)

        if field_name == "tactic":
            tactic = str(metadata.get("tactic", "")).strip().lower()
            return bool(tactic) and tactic == str(expected).strip().lower()

        if field_name == "asset":
            assets = metadata.get("assets", []) or []
            return str(expected) in [str(asset) for asset in assets]

        # Generic key lookup
        return metadata.get(field_name) == expected

    def _match_rule_window(
        self,
        rule: OntologyRule,
        window: Sequence[Tuple[str, Mapping[str, Any]]],
    ) -> bool:
        if len(rule.antecedent) != len(window):
            return False
        for condition, (tech, meta) in zip(rule.antecedent, window):
            if not self._conditions_hold(condition, tech, meta):
                return False
        return True

    def _infer_causal_links_from_rules(
        self, sequence: Sequence[str], metadata: Sequence[Mapping[str, Any]]
    ) -> List[CausalAnnotation]:
        annotations: List[CausalAnnotation] = []
        for rule in self.rules:
            window_size = len(rule.antecedent)
            if window_size == 0 or window_size > len(sequence):
                continue
            for idx in range(len(sequence) - window_size + 1):
                window = list(
                    zip(sequence[idx : idx + window_size], metadata[idx : idx + window_size])
                )
                if not self._match_rule_window(rule, window):
                    continue
                source = sequence[idx]
                target = sequence[idx + window_size - 1]
                relation = str(rule.consequent.get("relation", self.default_relation))
                annotations.append(
                    CausalAnnotation(
                        source=_normalize_tid(source),
                        target=_normalize_tid(target),
                        relation=relation,
                        rule=rule.name,
                        evidence={"window": [tech for tech, _ in window]},
                    )
                )
        return annotations

    def _infer_parallel_branches(
        self, sequence: Sequence[str], metadata: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        asset_to_indices: Dict[str, List[int]] = {}
        for idx, meta in enumerate(metadata):
            for asset in meta.get("assets", []) or []:
                asset_to_indices.setdefault(str(asset), []).append(idx)

        branches: List[Dict[str, Any]] = []
        for asset, indices in asset_to_indices.items():
            if len(indices) < 2:
                continue
            branches.append(
                {
                    "asset": asset,
                    "techniques": [sequence[i] for i in indices],
                    "indices": indices,
                }
            )
        return branches

    def apply(self, sequence: Sequence[str]) -> EnhancedSequence:
        techniques = list(sequence)
        metadata = [self._technique_metadata.get(_normalize_tid(tech), {}) for tech in techniques]

        causal_annotations: List[CausalAnnotation] = []
        for i, source in enumerate(techniques):
            for j in range(i + 1, len(techniques)):
                target = techniques[j]
                relation = self._relation_lookup.get(
                    (_normalize_tid(source), _normalize_tid(target))
                )
                if relation:
                    causal_annotations.append(
                        CausalAnnotation(
                            source=_normalize_tid(source),
                            target=_normalize_tid(target),
                            relation=relation,
                            rule="ontology_relation",
                            evidence={"offset": j - i},
                        )
                    )

        causal_annotations.extend(self._infer_causal_links_from_rules(techniques, metadata))
        parallel_branches = self._infer_parallel_branches(techniques, metadata)

        return EnhancedSequence(techniques, causal_annotations, parallel_branches)

    def estimate_causal_support(self, history: Sequence[str], candidate: str) -> float:
        if not history:
            return 0.0
        source = history[-1]
        meta_source = self._technique_metadata.get(_normalize_tid(source), {})
        meta_candidate = self._technique_metadata.get(_normalize_tid(candidate), {})

        relation = self._relation_lookup.get(
            (_normalize_tid(source), _normalize_tid(candidate))
        )
        if relation:
            return 1.0

        for rule in self.rules:
            if len(rule.antecedent) != 2:
                continue
            window = [(source, meta_source), (candidate, meta_candidate)]
            if self._match_rule_window(rule, window):
                return rule.weight
        return 0.0

    def persist_annotations(
        self,
        path: Path | str,
        sequence: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        result = self.apply(sequence)
        record = {
            "sequence": list(sequence),
            "metadata": dict(metadata or {}),
            "causal_annotations": [ann.as_dict() for ann in result.causal_annotations],
            "parallel_branches": result.parallel_branches,
        }

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if dest.exists():
            try:
                with open(dest, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(record)

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        return dest


__all__ = [
    "CausalAnnotation",
    "EnhancedSequence",
    "OntologyReasoner",
    "OntologyRule",
]
