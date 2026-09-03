"""
Unit tests for sigma_stealth.py module.
"""

import pytest
import yaml
import math
from pathlib import Path

from src.chain_paths.sigma_stealth import extract_attack_tags, compute_detect_score, count_detection_clauses


@pytest.fixture
def sample_sigma_rule():
    """Provides a sample Sigma rule for testing."""
    return """
title: Hypervisor Enforced Code Integrity Disabled
id: 8b7273a4-ba5d-4d8a-b04f-11f2900d043a
status: test
description: Test rule
tags:
    - attack.defense-evasion
    - attack.t1562.001
logsource:
    category: registry_set
    product: windows
detection:
    selection:
        TargetObject|endswith:
            - '\\DeviceGuard\\HypervisorEnforcedCodeIntegrity'
        Details: 'DWORD (0x00000000)'
    condition: selection
falsepositives:
    - Unknown
level: high
"""

def test_extract_attack_tags():
    """Tests the extraction of ATT&CK tags."""
    tags = ["attack.defense-evasion", "attack.t1562.001", "attack.t1059"]
    assert extract_attack_tags(tags) == ["T1562.001", "T1059"]
    assert extract_attack_tags(["attack.t1234.005"]) == ["T1234.005"]
    assert extract_attack_tags(["not.a.tag"]) == []


def test_compute_detect_score(sample_sigma_rule):
    """Tests the exact calculation of detect_score for a rule."""
    rule_doc = yaml.safe_load(sample_sigma_rule)

    # Manually calculate expected score based on the formula
    level_weight = 0.85  # high
    cond_complexity = 2  # TargetObject|endswith and Details
    fp_count = 1
    fp_factor = 1 / (1 + 0.5 * 1)
    expected_score = level_weight * math.log(1 + cond_complexity) * fp_factor
    
    score, metadata = compute_detect_score(rule_doc)
    assert score == pytest.approx(expected_score)
    assert metadata['level'] == 'high'
    assert metadata['cond_complexity'] == cond_complexity
    assert metadata['fp_count'] == fp_count