"""
Utilities for handling ATT&CK technique IDs.
"""

import re
from typing import Optional

import pandas as pd


def canonicalize_technique_id(tech_str: str) -> Optional[str]:
    """
    Canonicalize technique string to MITRE ATT&CK TID format.
    
    Args:
        tech_str: Raw technique string
        
    Returns:
        Canonical technique ID (T#### or T####.###) or None if invalid
    """
    if pd.isna(tech_str) or not isinstance(tech_str, str):
        return None
    
    # Clean the string
    tech_str = tech_str.strip().upper()
    
    # Pattern to match T#### or T####.### format
    pattern = r'T(\d{4}(?:\.\d+)?)'
    match = re.search(pattern, tech_str)
    
    if match:
        return f"T{match.group(1)}"
    
    # Try to extract numbers and construct TID
    numbers = re.findall(r'\d+', tech_str)
    if len(numbers) >= 1:
        base_num = numbers[0]
        if len(base_num) == 4:  # T####
            if len(numbers) >= 2 and len(numbers[1]) <= 3:  # T####.###
                return f"T{base_num}.{numbers[1]}"
            else:
                return f"T{base_num}"
    
    return None
