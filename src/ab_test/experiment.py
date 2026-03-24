"""
A/B experiment management for recommendation strategy testing.

Two strategies are tested in parallel:

* **Variant A — purchase_history_based**: Recommends items similar to what
  the buyer has previously bought.  Optimises for familiarity and conversion
  from proven preferences.

* **Variant B — browsing_pattern_based**: Recommends items that match the
  buyer's browsing-tag affinity but haven't been purchased yet.  Takes a
  discovery angle to surface new categories and expand basket size.

Assignment is deterministic: ``hash(user_id) % 2`` maps each user to a stable
variant for the lifetime of the experiment so metrics can be attributed cleanly.

Results are logged to ``ReviewStore`` so they join the same human-review
pipeline used elsewhere in the system.  Approved results can be promoted to
the golden evaluation set and used as few-shot examples for future prompt
improvements.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

import structlog

from src.agents.memory.buyer_profile import BuyerProfile
from src.api.review import ReviewStore


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Variant enum
# ---------------------------------------------------------------------------


class Variant(str, Enum):
    """Recommendation strategy variant.

    The string values are used as the ``reason`` label in ReviewStore so ops
    dashboards can filter experiment results without needing an enum lookup.
    """

    A = "purchase_history_based"
    B = "browsing_pattern_based"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_A = """\
You are writing a personalised product recommendation email.

Buyer profile:
- Top category of interest: {top_category}
- Previous purchases: {purchase_ids}

Recommend products similar to what this buyer has previously purchased. \
Focus on familiar styles, proven preferences, and complementary items. \
Keep the tone warm and confident.

Write a short, personalised recommendation email (3–5 sentences).\
"""

_PROMPT_B = """\
You are writing a personalised product discovery email.

Buyer profile:
- Top browsing interests (by tag): {top_tags}
- Top category: {top_category}

Recommend products that match this buyer's browsing patterns but that they \
haven't purchased yet. Take a discovery angle — introduce them to items that \
expand on their existing interests. Keep the tone enthusiastic and curious.

Write a short, engaging discovery email (3–5 sentences).\
"""


# ---------------------------------------------------------------------------
# ABExperiment
# ---------------------------------------------------------------------------


class ABExperiment:
    """Assign variants, build prompts, and log results for A/B analysis.

    Parameters
    ----------
    review_store:
        ReviewStore instance used to persist experiment results.
        Defaults to a fresh in-process store if not provided; pass an
        explicit store to share state across agent runs or to inject a
        test double.
    """

    def __init__(self, review_store: Optional[ReviewStore] = None) -> None:
        self._store = review_store or ReviewStore()

    # ------------------------------------------------------------------
    # Variant assignment
    # ------------------------------------------------------------------

    def assign_variant(self, user_id: str) -> Variant:
        """Map a user deterministically to a variant.

        Uses ``hash(user_id) % 2`` so every call with the same ``user_id``
        returns the same variant within a process.  The distribution is
        roughly 50/50 across a large user population due to hash uniformity.
        """
        bucket = hash(user_id) % 2
        variant = Variant.A if bucket == 0 else Variant.B
        logger.debug("ab_variant_assigned", user_id=user_id, variant=variant.value)
        return variant

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def get_prompt(self, variant: Variant, buyer_profile: BuyerProfile) -> str:
        """Return the email-generation prompt for the given variant and profile.

        Variant A emphasises purchase history and familiar styles.
        Variant B emphasises browsing-tag affinity and discovery.
        """
        top_category = buyer_profile["top_category"]

        if variant is Variant.A:
            purchase_ids = buyer_profile["purchase_history"]
            ids_str = ", ".join(purchase_ids[:5]) if purchase_ids else "none yet"
            return _PROMPT_A.format(
                top_category=top_category,
                purchase_ids=ids_str,
            )

        # Variant B — pick the top 3 affinity tags by score, highest first.
        tag_affinity: Dict[str, float] = buyer_profile["tag_affinity"]
        top_tags = sorted(tag_affinity, key=tag_affinity.get, reverse=True)[:3]  # type: ignore[arg-type]
        tags_str = ", ".join(top_tags) if top_tags else "various products"
        return _PROMPT_B.format(
            top_tags=tags_str,
            top_category=top_category,
        )

    # ------------------------------------------------------------------
    # Result logging
    # ------------------------------------------------------------------

    def log_result(
        self,
        user_id: str,
        variant: Variant,
        email_content: str,
        asset_ids: List[str],
    ) -> str:
        """Persist an experiment result to ReviewStore and return its item ID.

        The record structure mirrors how other pipeline outputs are stored so
        the same review UI and approval workflow can be used for experiment
        analysis without additional tooling.
        """
        item_id = self._store.add_item(
            record={
                "user_id": user_id,
                "variant": variant.value,
                "asset_ids": asset_ids,
            },
            output={"email": email_content},
            reason=f"ab_experiment:{variant.value}",
        )
        logger.info(
            "ab_result_logged",
            user_id=user_id,
            variant=variant.value,
            item_id=item_id,
            asset_count=len(asset_ids),
        )
        return item_id
