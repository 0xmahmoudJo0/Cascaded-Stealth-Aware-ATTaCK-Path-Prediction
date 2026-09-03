"""Lightweight preprocessing for parsed attack sequences."""
from .attack_sequence import AttackSequence


def preprocess_sequence(parsed_seq: AttackSequence) -> AttackSequence:
    """Normalize and validate parsed sequences.

    Currently trims empty tokens and normalizes casing, but can be extended
    with richer validation rules.
    """

    normalized = [token.strip().upper() for token in parsed_seq.path if token.strip()]
    metadata = {**parsed_seq.metadata, "preprocessed": True}
    return AttackSequence(
        path=normalized,
        metadata=metadata,
        branch_id=parsed_seq.branch_id,
        completed=parsed_seq.completed,
    )
