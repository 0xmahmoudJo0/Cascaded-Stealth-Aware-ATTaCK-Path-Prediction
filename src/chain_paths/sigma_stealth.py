"""
Sigma rule processing to compute stealth scores for techniques.
"""

import yaml
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

from . import config as cfg
from .io import save_csv, save_json, ensure_dir


def extract_attack_tags(tags: List[str]) -> List[str]:
    """Extracts canonical ATT&CK technique IDs from Sigma rule tags."""
    if not tags:
        return []
    
    attack_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower.startswith('attack.t'):
            try:
                # Handles both T1234 and T1234.001
                tid = 'T' + tag.split('.t')[-1]
                if tid.count('.') > 1: # Avoid invalid tags like attack.t1102.001.002
                    continue
                attack_tags.append(tid.upper())
            except:
                continue
    return attack_tags


def get_level_weight(level: str) -> float:
    """Maps Sigma rule level to a numeric weight."""
    level_map = {'critical': 1.0, 'high': 0.85, 'medium': 0.5, 'low': 0.25}
    return level_map.get(level.lower(), 0.4)


def count_detection_clauses(detection: Any) -> int:
    """Recursively counts clauses in a detection dictionary."""
    if isinstance(detection, dict):
        # Exclude 'condition' key from count, sum counts of values
        return sum(count_detection_clauses(v) for k, v in detection.items() if k != 'condition')
    elif isinstance(detection, list):
        # Sum counts for each item in the list
        return sum(count_detection_clauses(item) for item in detection)
    else:
        # Leaf node
        return 1


def compute_detect_score(rule_doc: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Computes the detection score for a single Sigma rule."""
    level = rule_doc.get('level', 'informational')
    level_weight = get_level_weight(level)
    
    detection = rule_doc.get('detection', {})
    cond_complexity = count_detection_clauses(detection)
    
    false_positives = rule_doc.get('falsepositives', [])
    fp_count = len(false_positives)
    fp_factor = 1 / (1 + 0.5 * fp_count)
    
    detect_score = level_weight * math.log(1 + cond_complexity) * fp_factor
    
    metadata = {
        'title': rule_doc.get('title'),
        'level': level,
        'level_weight': level_weight,
        'cond_complexity': cond_complexity,
        'fp_count': fp_count,
        'fp_factor': fp_factor,
        'detect_score': detect_score,
        'tags': rule_doc.get('tags', [])
    }
    return detect_score, metadata


def build_sigma_stealth_csv(rules_dir: Path, output_csv: Path, details_json: Path):
    """
    Processes all Sigma rules to build the sigma_stealth.csv file.
    """
    print(f"Processing Sigma rules from: {rules_dir}")
    tech_scores = defaultdict(float)
    tech_rule_counts = defaultdict(int)
    rule_details = []

    rule_files = list(rules_dir.rglob('*.yml'))
    for rule_path in tqdm(rule_files, desc="Processing Sigma rules"):
        try:
            with open(rule_path, 'r', encoding='utf-8') as f:
                for rule_doc in yaml.safe_load_all(f):
                    if not rule_doc or not isinstance(rule_doc, dict):
                        continue
                    
                    tags = extract_attack_tags(rule_doc.get('tags', []))
                    if not tags:
                        continue

                    score, metadata = compute_detect_score(rule_doc)
                    metadata['file_path'] = str(rule_path.relative_to(rules_dir.parent.parent))
                    rule_details.append(metadata)

                    for tid in tags:
                        tech_scores[tid] += score
                        tech_rule_counts[tid] += 1
        except Exception as e:
            print(f"Warning: Could not process file {rule_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame({
        'technique_id': list(tech_scores.keys()),
        'D_raw': list(tech_scores.values()),
        'rule_count': [tech_rule_counts[tid] for tid in tech_scores.keys()]
    })

    # Normalize and compute stealth score
    max_d_raw = df['D_raw'].max()
    df['D_norm'] = df['D_raw'] / max_d_raw if max_d_raw > 0 else 0
    
    beta = cfg.DEFAULT_PARAMS.get('beta', 1.0)
    df['stealth_S'] = 1 / (1 + np.log(1 + beta * df['D_norm']))

    # Save files
    ensure_dir(output_csv.parent)
    save_csv(df, output_csv)
    print(f"Saved stealth scores to {output_csv}")

    ensure_dir(details_json.parent)
    save_json(rule_details, details_json)
    print(f"Saved rule details to {details_json}")