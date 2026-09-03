import json
from pathlib import Path

import pytest

from src.chain_paths.ontology_reasoning import OntologyReasoner, OntologyRule


def test_apply_ontology_reasoning_inferrs_links_and_parallel_branches(tmp_path: Path):
    ontology = {
        "techniques": {
            "T1001": {"tactic": "reconnaissance", "assets": ["host"]},
            "T3000": {"tactic": "exfiltration", "assets": ["db"]},
            "T2000": {"tactic": "impact", "assets": ["db", "host"]},
        },
        "relations": [
            {"source": "T3000", "target": "T2000", "relation": "causes"}
        ],
    }
    rules = [
        OntologyRule(
            name="exfil_to_impact",
            antecedent=[
                {"field": "tactic", "equals": "exfiltration"},
                {"field": "tactic", "equals": "impact"},
            ],
            consequent={"relation": "causes"},
            weight=0.75,
        )
    ]

    reasoner = OntologyReasoner(ontology, rules)
    sequence = ["T1001", "T3000", "T2000"]
    enhanced = reasoner.apply(sequence)

    causal_pairs = {(ann.source, ann.target) for ann in enhanced.causal_annotations}
    assert ("T3000", "T2000") in causal_pairs
    parallel_assets = {branch["asset"] for branch in enhanced.parallel_branches}
    assert "db" in parallel_assets
    db_branch = next(branch for branch in enhanced.parallel_branches if branch["asset"] == "db")
    assert set(db_branch["techniques"]) == {"T3000", "T2000"}

    path = reasoner.persist_annotations(tmp_path / "annotations.json", sequence)
    stored = json.loads(path.read_text())
    assert stored[-1]["causal_annotations"]


def test_causal_support_scores_last_hop():
    ontology = {
        "techniques": {
            "T1111": {"tactic": "collection", "assets": ["db"]},
            "T2222": {"tactic": "exfiltration", "assets": ["db"]},
        },
    }
    rules = [
        OntologyRule(
            name="collection_to_exfil",
            antecedent=[
                {"field": "technique", "equals": "T1111"},
                {"field": "technique", "equals": "T2222"},
            ],
            consequent={"relation": "causes"},
            weight=0.5,
        )
    ]
    reasoner = OntologyReasoner(ontology, rules)

    assert reasoner.estimate_causal_support(["T1111"], "T2222") == pytest.approx(0.5)
    assert reasoner.estimate_causal_support([], "T2222") == 0.0
