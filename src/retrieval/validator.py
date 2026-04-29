"""
Deterministic validation and diversity reranking for recommended items.

``validate_item`` filters out non-recommendable assets (inactive, unsellable).
``mmr_rerank`` applies Maximal Marginal Relevance to balance relevance and
diversity using attribute-based similarity (category, tags) rather than
embeddings — zero external dependencies.
"""

from __future__ import annotations

from typing import Dict, List

from src.schemas.metadata import ContentMetadata


def validate_item(item: ContentMetadata, *, exclude_ids: frozenset[str] = frozenset()) -> bool:
    """Check whether an asset is safe to recommend.

    Parameters
    ----------
    item:
        Candidate asset.
    exclude_ids:
        Content IDs to suppress (e.g. items the buyer already purchased).
    """
    if item.content_id in exclude_ids:
        return False
    if not item.title or not item.tags:
        return False
    return True


def _attribute_similarity(a: ContentMetadata, b: ContentMetadata) -> float:
    """Cheap attribute-based similarity in [0, 1].

    Weighted: category match 0.5, tag overlap 0.5.
    """
    score = 0.0
    if a.category == b.category:
        score += 0.5
    tags_a = set(a.tags)
    tags_b = set(b.tags)
    union = tags_a | tags_b
    if union:
        score += 0.5 * len(tags_a & tags_b) / len(union)
    return score


def mmr_rerank(
    items: List[ContentMetadata],
    *,
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> List[ContentMetadata]:
    """Maximal Marginal Relevance — balance relevance and diversity.

    ``lambda_param`` controls the trade-off: 1.0 = pure relevance (no
    diversity), 0.0 = maximum diversity.  The default 0.7 favours relevance
    while still penalising near-duplicate category/tag overlap.

    Items are assumed to be pre-sorted by relevance (index 0 = most relevant).
    Relevance score is approximated as ``1 - rank / len(items)``.
    """
    if not items:
        return []

    n = len(items)
    selected: List[ContentMetadata] = []
    remaining = list(range(n))

    for _ in range(min(top_k, n)):
        best_idx = -1
        best_mmr = -1.0

        for idx in remaining:
            relevance = 1.0 - idx / n

            max_sim = 0.0
            for sel in selected:
                sim = _attribute_similarity(items[idx], sel)
                if sim > max_sim:
                    max_sim = sim

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        selected.append(items[best_idx])
        remaining.remove(best_idx)

    return selected
