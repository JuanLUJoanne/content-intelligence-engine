"""
In-memory asset retrieval with tag-affinity scoring.

AssetRetriever ranks a static corpus of ContentMetadata objects against a
free-text query plus structured filters.  No vector store is needed at this
stage: tag overlap already carries the signal required for personalised
recommendations because the upstream pipeline ensures tags are normalised
(lowercase, de-duplicated) and domain-consistent.

Scoring model
-------------
Each candidate asset receives a score:

    score = Σ query_term_hits × 1.0
           + Σ filter_tag_hits × 1.0
           + category_match × 2.0      (structured, high-confidence signal)
           + condition_match × 0.5     (weak preference signal)

Query terms are split on whitespace and matched against asset tags.  Filters
let callers add hard affinity signals (e.g. "must be electronics") without
polluting the free-text query.  Category is weighted higher than tags because
it is a single, deliberate classification rather than a noisy free-text label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)

_CATEGORY_MATCH_SCORE: float = 2.0
_TAG_MATCH_SCORE: float = 1.0
_CONDITION_MATCH_SCORE: float = 0.5


class AssetRetriever:
    """Rank and return the top-N assets from an in-memory corpus.

    Instantiate once with a corpus snapshot and call ``search()`` as many
    times as needed.  The corpus is treated as immutable after construction;
    callers who need to refresh it should create a new retriever.
    """

    def __init__(
        self,
        corpus: List[ContentMetadata],
        *,
        top_n: int = 5,
    ) -> None:
        self._corpus = list(corpus)
        self._top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ContentMetadata]:
        """Return the top-N assets ranked by affinity to ``query`` and ``filters``.

        Parameters
        ----------
        query:
            Free-text search terms, space-separated.  Each term is matched
            against the asset's normalised tag list.
        filters:
            Optional structured constraints:

            * ``"tags"`` — list of tag strings that must overlap with asset tags
              (each match adds to the score, not a hard filter).
            * ``"category"`` — category value string; matching assets get a
              bonus of ``_CATEGORY_MATCH_SCORE``.
            * ``"condition"`` — condition value string; matching assets get a
              small bonus.

        Returns
        -------
        List[ContentMetadata]
            Up to ``top_n`` assets ordered from highest to lowest score.
            Assets with score 0 are still returned if the corpus has fewer
            than ``top_n`` items, so callers can always expect a non-empty
            result when the corpus is non-empty.
        """
        filters = filters or {}
        query_terms = {t for t in query.lower().split() if t}
        filter_tags = {t.lower() for t in filters.get("tags", [])}
        filter_category: Optional[str] = filters.get("category")
        filter_condition: Optional[str] = filters.get("condition")

        scored: List[Tuple[float, ContentMetadata]] = []
        for asset in self._corpus:
            score = self._score_asset(
                asset,
                query_terms=query_terms,
                filter_tags=filter_tags,
                filter_category=filter_category,
                filter_condition=filter_condition,
            )
            scored.append((score, asset))

        # Stable sort: primary key is score (descending), secondary is title
        # (ascending) so results are deterministic when scores tie.
        scored.sort(key=lambda x: (-x[0], x[1].title))

        results = [asset for _, asset in scored[: self._top_n]]

        logger.info(
            "asset_retriever_search",
            query=query,
            filter_category=filter_category,
            filter_tags=list(filter_tags),
            result_count=len(results),
            top_score=scored[0][0] if scored else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_asset(
        self,
        asset: ContentMetadata,
        *,
        query_terms: set[str],
        filter_tags: set[str],
        filter_category: Optional[str],
        filter_condition: Optional[str],
    ) -> float:
        asset_tags = set(asset.tags)
        score = 0.0

        # Free-text query terms matched against asset tags.
        score += sum(_TAG_MATCH_SCORE for t in query_terms if t in asset_tags)

        # Caller-supplied tag filter — additive, not exclusive.
        score += sum(_TAG_MATCH_SCORE for t in filter_tags if t in asset_tags)

        # Structured category signal — carries more weight than tag noise.
        if filter_category and asset.category.value == filter_category:
            score += _CATEGORY_MATCH_SCORE

        # Condition is a weak preference signal.
        if filter_condition and asset.condition.value == filter_condition:
            score += _CONDITION_MATCH_SCORE

        return score
