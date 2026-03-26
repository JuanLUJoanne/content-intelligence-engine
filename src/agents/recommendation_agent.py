"""
Recommendation agent: agentic pipeline for personalised product emails.

Pipeline
--------
1. Build BuyerProfile from purchase and browsing history.
2. Assign A/B variant (deterministic, hash-based).
3. Adaptive retrieval — LLM-in-the-loop search with up to 3 refinement
   rounds.  The LLM observes search results, decides if they are relevant,
   and can call ``search_assets`` again with a different query.  Falls back
   to single-shot retrieval when the LLM is a DummyProvider.
4. Generate recommendation email via LLM provider.
5. Record cost via CostTracker.
6. Evaluate email quality via LLMJudge.
7. Log result to ABExperiment (persisted in ReviewStore).
8. Return RecommendationResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

import structlog

from src.ab_test.experiment import ABExperiment, Variant
from src.agents.adaptive_retriever import AdaptiveRetriever
from src.agents.memory.buyer_profile import BuyerProfile, build_buyer_profile, make_buyer_profile
from src.eval.judge import LLMJudge
from src.gateway.cost_tracker import CostTracker
from src.gateway.providers import LLMProvider, ProviderFactory
from src.mcp.client import MCPClient
from src.retrieval.asset_retriever import AssetRetriever
from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RecommendationResult:
    """Output of a single agent run.

    Attributes
    ----------
    email:
        The generated recommendation email text.
    assets:
        The ranked assets included in the recommendation (up to 5).
    variant:
        The A/B variant used to generate this recommendation.
    eval_score:
        Overall quality score (0–1) from LLMJudge across all five eval
        dimensions.  Higher is better.
    cost:
        Total LLM spend for this run in the unit of the CostTracker's pricing
        (typically USD).  Zero when no CostTracker is provided.
    """

    email: str
    assets: List[ContentMetadata]
    variant: Variant
    eval_score: float
    cost: Decimal


# ---------------------------------------------------------------------------
# RecommendationAgent
# ---------------------------------------------------------------------------


class RecommendationAgent:
    """Orchestrates the recommendation pipeline for a single user.

    All dependencies are injectable so tests can swap in stubs without
    patching module globals.  Sensible defaults are provided for every
    optional parameter so the agent can be used with zero configuration
    in development and CI.

    Parameters
    ----------
    model_id:
        LLM model identifier passed to the provider and LLMJudge.
    cost_tracker:
        Optional CostTracker; if omitted, cost is reported as Decimal("0").
    judge:
        Optional LLMJudge; defaults to ``LLMJudge.from_provider(model_id)``.
    experiment:
        Optional ABExperiment; defaults to a fresh instance with a default
        ReviewStore.
    """

    _TOP_N_ASSETS = 5
    _TOP_AFFINITY_TAGS = 3

    def __init__(
        self,
        *,
        model_id: str = "dummy",
        cost_tracker: Optional[CostTracker] = None,
        judge: Optional[LLMJudge] = None,
        experiment: Optional[ABExperiment] = None,
    ) -> None:
        self._model_id = model_id
        self._cost_tracker = cost_tracker
        self._judge = judge or LLMJudge.from_provider(model_id)
        self._experiment = experiment or ABExperiment()
        self._provider = ProviderFactory.get_provider(model_id)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str,
        available_assets: List[ContentMetadata],
        *,
        purchased_assets: Optional[List[ContentMetadata]] = None,
        browsed_assets: Optional[List[ContentMetadata]] = None,
        mcp_client: Optional[MCPClient] = None,
    ) -> RecommendationResult:
        """Run the full recommendation pipeline and return a result.

        Parameters
        ----------
        user_id:
            Stable buyer identifier used for variant assignment and logging.
        available_assets:
            Full corpus of assets the agent may recommend from.
        purchased_assets:
            Resolved ContentMetadata objects for the buyer's purchase history.
            Ignored when ``mcp_client`` is provided.  Defaults to an empty list
            (new buyer).
        browsed_assets:
            Resolved ContentMetadata objects for the buyer's browsing history.
            Ignored when ``mcp_client`` is provided.  Defaults to the first few
            available assets when empty so the agent always has *some* affinity
            signal to work with.
        mcp_client:
            Optional MCPClient.  When provided the agent fetches buyer history
            via MCP instead of using the ``purchased_assets``/``browsed_assets``
            arguments.  Supports both mock (``use_mock=True``) and real
            (``use_mock=False``) data sources.
        """
        # Step 1 — build buyer profile
        if mcp_client is not None:
            profile = await build_buyer_profile(user_id, mcp_client)
        else:
            purchased = purchased_assets or []
            browsed = browsed_assets or available_assets[: self._TOP_AFFINITY_TAGS]
            profile = make_buyer_profile(user_id, purchased, browsed)

        # Step 2 — assign variant
        variant = self._experiment.assign_variant(user_id)

        # Step 3 — adaptive retrieval (LLM-in-the-loop)
        assets = await self._retrieve(available_assets, profile, variant)

        # Step 4 — generate email
        email, prompt_used = await self._generate_email(profile, variant, assets)

        # Step 5 — record cost
        cost = self._record_cost(prompt_used)

        # Step 6 — evaluate quality
        eval_result = await self._judge.score(
            user_input=prompt_used,
            candidate_output=email,
        )

        # Step 7 — log experiment result
        self._experiment.log_result(
            user_id,
            variant,
            email,
            [a.content_id for a in assets],
        )

        logger.info(
            "recommendation_complete",
            user_id=user_id,
            variant=variant.value,
            eval_score=eval_result.overall_score,
            asset_count=len(assets),
            cost=str(cost),
        )

        return RecommendationResult(
            email=email,
            assets=assets,
            variant=variant,
            eval_score=eval_result.overall_score,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        corpus: List[ContentMetadata],
        profile: BuyerProfile,
        variant: Variant,
    ) -> List[ContentMetadata]:
        """Run adaptive retrieval: variant determines the initial query,
        then the LLM decides whether to refine."""
        base_retriever = AssetRetriever(corpus, top_n=self._TOP_N_ASSETS)
        adaptive = AdaptiveRetriever(
            base_retriever,
            self._provider,
            model_id=self._model_id,
            max_rounds=3,
            top_n=self._TOP_N_ASSETS,
        )

        if variant is Variant.A:
            query = profile["top_category"]
            filters: Optional[Dict] = {"category": profile["top_category"]}
        else:
            tag_affinity = profile["tag_affinity"]
            top_tags = sorted(
                tag_affinity, key=tag_affinity.get, reverse=True  # type: ignore[arg-type]
            )[: self._TOP_AFFINITY_TAGS]
            query = " ".join(top_tags) if top_tags else profile["top_category"]
            filters = {"tags": top_tags}

        return await adaptive.search(
            profile, initial_query=query, initial_filters=filters,
        )

    async def _generate_email(
        self,
        profile: BuyerProfile,
        variant: Variant,
        assets: List[ContentMetadata],
    ) -> tuple[str, str]:
        """Build the full prompt, call the provider, and return (email, prompt).

        The provider is expected to return a dict (consistent with the rest of
        the pipeline).  We extract a ``"title"`` or ``"description"`` field as
        the email body; if neither is present we serialise the whole response
        so something meaningful is always returned.
        """
        base_prompt = self._experiment.get_prompt(variant, profile)
        asset_lines = "\n".join(
            f"- {a.title} ({a.category.value}, {a.price_range.value}): "
            + ", ".join(a.tags[:3])
            for a in assets
        )
        full_prompt = (
            f"{base_prompt}\n\n"
            f"Products to recommend:\n{asset_lines}"
        )

        raw = await self._provider.generate(full_prompt, self._model_id)

        # DummyProvider and real providers both return dicts; pull the most
        # readable text field available.
        if isinstance(raw, dict):
            email = (
                raw.get("description")
                or raw.get("title")
                or json.dumps(raw, ensure_ascii=False)
            )
        else:
            email = str(raw)

        logger.info(
            "recommendation_email_generated",
            variant=variant.value,
            prompt_length=len(full_prompt),
        )
        return email, full_prompt

    def _record_cost(self, prompt: str) -> Decimal:
        """Record LLM cost and return it; returns Decimal('0') gracefully on failure."""
        if self._cost_tracker is None:
            return Decimal("0")

        input_tokens = max(len(prompt) // 4, 1)
        output_tokens = 128  # conservative estimate for a short email

        try:
            return self._cost_tracker.record_usage(
                self._model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            logger.warning(
                "recommendation_cost_record_failed",
                model=self._model_id,
                error=str(exc),
            )
            return Decimal("0")
