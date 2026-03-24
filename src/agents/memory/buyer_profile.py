"""
Buyer profile construction and tag-affinity computation.

A BuyerProfile is a lightweight snapshot of what a buyer has bought and
browsed.  It is the primary input for variant-specific retrieval: variant A
uses ``top_category`` (derived from purchase history) to find familiar items,
while variant B uses ``tag_affinity`` (derived from browsing history) to
surface discovery-angle recommendations.

Keeping profile construction here — away from the agent orchestration layer —
makes it easy to swap in a real user-service lookup later without touching
retrieval or prompt logic.
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# BuyerProfile TypedDict
# ---------------------------------------------------------------------------


class BuyerProfile(dict):
    """Typed view of a buyer's behavioural history.

    Using a TypedDict-style class (backed by plain dict) keeps the shape
    explicit while remaining JSON-serialisable and easy to pass between
    agent steps without extra conversion.

    Fields
    ------
    user_id
        Stable identifier for the buyer.
    purchase_history
        Ordered list of asset IDs the buyer has previously purchased,
        most recent last.
    browsing_history
        Ordered list of asset IDs the buyer has viewed but not purchased.
    tag_affinity
        Normalised (0–1) affinity scores computed from browsing history.
        Higher score → buyer has browsed more assets with that tag.
    top_category
        Single category value that appears most frequently across the
        buyer's *purchase* history (falls back to browsing history if no
        purchases exist).
    """

    # Slot-style class variable so IDEs can surface the expected keys.
    __annotations__ = {
        "user_id": str,
        "purchase_history": List[str],
        "browsing_history": List[str],
        "tag_affinity": Dict[str, float],
        "top_category": str,
    }


# ---------------------------------------------------------------------------
# Tag-affinity computation
# ---------------------------------------------------------------------------


def compute_tag_affinity(
    browsing_history: List[ContentMetadata],
) -> Dict[str, float]:
    """Count tag frequency across browsed assets and normalise to [0, 1].

    Each tag on each browsed asset contributes one count.  The resulting
    raw counts are divided by the maximum count so the hottest tag always
    scores 1.0 and rarer tags are proportionally lower.  This makes affinity
    scores comparable across buyers with very different browsing volumes.

    Returns an empty dict when ``browsing_history`` is empty.
    """
    counts: Dict[str, int] = {}
    for asset in browsing_history:
        for tag in asset.tags:
            counts[tag] = counts.get(tag, 0) + 1

    if not counts:
        return {}

    max_count = max(counts.values())
    return {tag: count / max_count for tag, count in counts.items()}


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------


def _infer_top_category(assets: List[ContentMetadata]) -> str:
    """Return the most-browsed or most-purchased category value.

    Uses a simple frequency count; ties are broken by the natural ordering
    of category enum values (alphabetical) so the result is stable.
    Falls back to ``"electronics"`` when the asset list is empty.
    """
    if not assets:
        return "electronics"

    counts: Dict[str, int] = {}
    for asset in assets:
        cat = asset.category.value
        counts[cat] = counts.get(cat, 0) + 1

    return max(counts, key=lambda k: (counts[k], k))


# ---------------------------------------------------------------------------
# Profile factory
# ---------------------------------------------------------------------------


def make_buyer_profile(
    user_id: str,
    purchased_assets: List[ContentMetadata],
    browsed_assets: List[ContentMetadata],
) -> BuyerProfile:
    """Construct a BuyerProfile from resolved asset objects.

    ``purchased_assets`` and ``browsed_assets`` are the *resolved* asset
    objects, not raw IDs.  The caller is responsible for the lookup; this
    function only computes derived fields (tag affinity, top category) and
    packs everything into a BuyerProfile dict.
    """
    tag_affinity = compute_tag_affinity(browsed_assets)
    # Prefer purchase history for top_category; fall back to browsed assets
    # so new buyers with no purchases still get a meaningful category signal.
    top_category = _infer_top_category(purchased_assets or browsed_assets)

    profile = BuyerProfile(
        user_id=user_id,
        purchase_history=[a.content_id for a in purchased_assets],
        browsing_history=[a.content_id for a in browsed_assets],
        tag_affinity=tag_affinity,
        top_category=top_category,
    )

    logger.info(
        "buyer_profile_built",
        user_id=user_id,
        top_category=top_category,
        affinity_tags=len(tag_affinity),
        purchase_count=len(purchased_assets),
        browse_count=len(browsed_assets),
    )
    return profile
