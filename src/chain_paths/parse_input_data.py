"""Parse raw user inputs into :class:`AttackSequence` instances."""
from typing import Iterable, Union

from .attack_sequence import AttackSequence


def parse_input_data(input_data: Union[str, Iterable[str]]) -> AttackSequence:
    """Convert raw CLI inputs into an :class:`AttackSequence`.

    Args:
        input_data: Comma-separated string or iterable of technique IDs.

    Returns:
        AttackSequence seeded with normalized technique IDs.
    """

    if isinstance(input_data, str):
        seeds = [item.strip() for item in input_data.split(',') if item.strip()]
    else:
        seeds = [str(item).strip() for item in input_data if str(item).strip()]

    normalized = [seed.upper() for seed in seeds]
    return AttackSequence(path=normalized, metadata={"raw_input": input_data})
