"""
Trending items tool — deterministic fallback for cold-start users.

When a buyer has no purchase or browsing history the recommendation agent
needs a starting point.  ``TrendingTool`` returns mock trending items per
category using hash-based determinism (same as ``MCPClient``), so results
are reproducible in tests without fixtures or patching.
"""

from __future__ import annotations

from typing import List, Optional

from src.schemas.metadata import ContentMetadata, Category, Condition, PriceRange


# Pre-built pool of trending items per category.  In production this would
# be a time-windowed popularity query against the analytics store.
_TRENDING_POOL: dict[str, list[dict[str, str]]] = {
    "electronics": [
        {"title": "Wireless Noise-Cancelling Headphones", "tags": "headphones,wireless,audio"},
        {"title": "Smart Home Hub Controller", "tags": "smart_home,iot,controller"},
        {"title": "Portable Power Bank 20000mAh", "tags": "power_bank,portable,charging"},
        {"title": "Mechanical Gaming Keyboard", "tags": "keyboard,gaming,mechanical"},
    ],
    "clothing": [
        {"title": "Organic Cotton Oversized Tee", "tags": "cotton,casual,sustainable"},
        {"title": "Slim Fit Stretch Chinos", "tags": "chinos,slim_fit,workwear"},
        {"title": "Merino Wool Quarter-Zip", "tags": "merino,knitwear,layering"},
    ],
    "home": [
        {"title": "Minimalist Desk Lamp LED", "tags": "lamp,led,minimalist"},
        {"title": "Bamboo Kitchen Organiser Set", "tags": "bamboo,kitchen,organiser"},
        {"title": "Memory Foam Seat Cushion", "tags": "cushion,ergonomic,office"},
    ],
    "sports": [
        {"title": "Resistance Band Set (5-Pack)", "tags": "resistance,fitness,bands"},
        {"title": "Insulated Water Bottle 1L", "tags": "bottle,insulated,hydration"},
    ],
    "books": [
        {"title": "Designing Data-Intensive Applications", "tags": "engineering,distributed,databases"},
        {"title": "The Pragmatic Programmer", "tags": "engineering,craft,career"},
    ],
}


class TrendingTool:
    """Return trending items per category with deterministic output."""

    def get_trending(
        self,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[ContentMetadata]:
        """Return up to ``top_k`` trending items, optionally filtered by category.

        When ``category`` is ``None``, returns items across all categories.
        """
        if category and category in _TRENDING_POOL:
            pool = _TRENDING_POOL[category]
        else:
            pool = [item for items in _TRENDING_POOL.values() for item in items]

        results: List[ContentMetadata] = []
        for i, entry in enumerate(pool[:top_k]):
            cat = category or list(_TRENDING_POOL.keys())[i % len(_TRENDING_POOL)]
            results.append(
                ContentMetadata(
                    content_id=f"trending_{cat}_{i}",
                    title=entry["title"],
                    category=Category(cat) if cat in Category._value2member_map_ else Category.ELECTRONICS,
                    condition=Condition.NEW,
                    price_range=PriceRange.MID_RANGE,
                    tags=entry["tags"].split(","),
                )
            )
        return results
