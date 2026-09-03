"""Candidate generation utilities for base predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

import pandas as pd

from .counts import get_top_candidates_for_context


@dataclass
class CandidateConfig:
    """Configuration for candidate generation."""

    k_neighbors: int = 50
    include_unigram_top: int = 50
    max_candidates: int = 200


class CandidateGenerator:
    """Collect candidate techniques for scoring.

    The generator combines information from n-gram counts and embedding
    neighbours to build a deduplicated candidate pool.  It purposely keeps the
    implementation lightweight so both classical and neural base predictors can
    reuse it without depending on the n-gram model internals.
    """

    def __init__(
        self,
        counts_data: dict[int, pd.DataFrame] | None = None,
        *,
        technique_ids: Sequence[str] | None = None,
        unigram_fallback: Iterable[str] | None = None,
        embedding_lookup: callable | None = None,
        config: CandidateConfig | None = None,
    ) -> None:
        self.counts_data = counts_data or {}
        self.technique_ids = list(technique_ids or [])
        self.unigram_fallback = list(unigram_fallback or [])
        self.embedding_lookup = embedding_lookup
        self.config = config or CandidateConfig()

    # ------------------------------------------------------------------
    def _counts_candidates(self, context: Sequence[str]) -> List[str]:
        candidates: Set[str] = set()
        if not self.counts_data:
            return []

        if len(context) >= 2 and 3 in self.counts_data:
            trigram_ctx = list(context[-2:])
            candidates.update(
                get_top_candidates_for_context(trigram_ctx, self.counts_data[3], self.config.include_unigram_top)
            )
        if len(context) >= 1 and 2 in self.counts_data:
            bigram_ctx = list(context[-1:])
            candidates.update(
                get_top_candidates_for_context(bigram_ctx, self.counts_data[2], self.config.include_unigram_top)
            )
        if 1 in self.counts_data:
            unigram_df = self.counts_data[1]
            top_unigrams = (
                unigram_df.nlargest(self.config.include_unigram_top, 'count')['candidate'].tolist()
            )
            candidates.update(top_unigrams)
        return list(candidates)

    def _embedding_candidates(self, history: Sequence[str]) -> List[str]:
        if self.embedding_lookup is None or not history:
            return []
        return list(self.embedding_lookup(history, self.config.k_neighbors))

    def _fallback_candidates(self) -> List[str]:
        if self.unigram_fallback:
            return list(self.unigram_fallback)
        return self.technique_ids

    # ------------------------------------------------------------------
    def get_candidates(self, history: Sequence[str], top_n: int | None = None) -> List[str]:
        """Return a deduplicated list of candidate techniques."""

        top_n = top_n or self.config.max_candidates
        context = list(history)
        candidates: List[str] = []
        seen: Set[str] = set()

        def extend(items: Iterable[str]) -> None:
            for item in items:
                if item not in seen:
                    seen.add(item)
                    candidates.append(item)
                    if len(candidates) >= top_n:
                        return

        extend(self._counts_candidates(context))
        if len(candidates) < top_n:
            extend(self._embedding_candidates(context))
        if len(candidates) < top_n:
            extend(self._fallback_candidates())
        return candidates[:top_n]


__all__ = ["CandidateGenerator", "CandidateConfig"]
