"""
Heuristic router for choosing the right model per request.

The router exists to make routing decisions explicit and testable instead of
hard‑coding model names at call sites. By funnelling all traffic through a
single component that understands cost, complexity, and health, we can evolve
the fleet (and tame spend) without rewriting business logic.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import structlog

from .circuit_breaker import CircuitBreaker, CircuitState
from .cost_tracker import CostTracker


logger = structlog.get_logger(__name__)


class ModelTier(str, Enum):
    """Coarse performance/cost buckets for models.

    Tiers let us express routing intentions in business language (e.g. "flash"
    for cheap bulk processing vs "premium" for critical flows) while retaining
    the freedom to swap out concrete model IDs underneath.
    """

    FLASH = "flash"
    STANDARD = "standard"
    PREMIUM = "premium"
    LOCAL = "local"


@dataclass(frozen=True)
class TaskFeatures:
    """Compact description of a request used for routing decisions.

    We keep this as a separate type so new features (like tool‑use, batch
    size, tenant priority) can be added over time without changing public
    router APIs or leaking low‑level details to callers.
    """

    estimated_input_tokens: int
    estimated_output_tokens: int
    latency_sensitivity: float  # 0.0 (can wait) .. 1.0 (needs fastest)
    quality_sensitivity: float  # 0.0 (ok to degrade) .. 1.0 (business‑critical)
    cost_sensitivity: float  # 0.0 (money no object) .. 1.0 (aggressive saving)
    allow_local_fallback: bool = True
    minimum_tier: ModelTier = ModelTier.FLASH


class NoAvailableModelError(RuntimeError):
    """Raised when the router cannot find any viable backend.

    This is intentionally loud because falling back to "whatever is left"
    silently would make it very hard to debug why large batches suddenly
    started failing or under‑performing.
    """


class ModelRouter:
    """Decision point for mapping requests to concrete models.

    The router coordinates three concerns: how "hard" a task is, how expensive
    each model would be, and whether individual backends are currently healthy.
    Keeping this logic in one place gives us a single knob to tune when we
    adjust budgets, roll out new models, or respond to vendor incidents.
    """

    def __init__(
        self,
        *,
        cost_tracker: CostTracker,
    ) -> None:
        self._cost_tracker = cost_tracker
        self._tiers_by_model: Dict[str, ModelTier] = {}
        self._breakers_by_model: Dict[str, CircuitBreaker] = {}
        # Simple lock to guard concurrent registration from async callers.
        self._lock = asyncio.Lock()

        # Tier ordering from most capable/expensive to least. Keeping this
        # central avoids scattering implicit assumptions about "higher" tiers.
        self._tier_order: Sequence[ModelTier] = (
            ModelTier.PREMIUM,
            ModelTier.STANDARD,
            ModelTier.FLASH,
        )

    async def register_model(
        self,
        model: str,
        tier: ModelTier,
        breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        """Register (or update) a model in the routing table.

        Using an async method here keeps us free to evolve registration to
        consult remote configuration or service discovery without changing
        call sites.
        """

        async with self._lock:
            self._tiers_by_model[model] = tier
            if breaker is not None:
                self._breakers_by_model[model] = breaker

    def _complexity_score(self, features: TaskFeatures) -> float:
        """Return a 0‑1 estimate of how demanding the task is.

        The scoring intentionally stays heuristic and monotonic rather than
        trying to be "perfect": the goal is to produce stable, explainable
        thresholds rather than chase marginal optimality.
        """

        tokens = max(
            features.estimated_input_tokens + features.estimated_output_tokens, 1
        )
        # Normalise token count into roughly 0..1 using a log curve so huge
        # outliers do not dominate every other signal.
        token_component = math.log10(tokens) / math.log10(16000)
        token_component = max(0.0, min(token_component, 1.0))

        # When quality absolutely matters we bias towards higher tiers even if
        # the token footprint is small.
        quality_component = features.quality_sensitivity

        # Latency sensitivity also nudges us up: faster models tend to cost
        # more, so we deliberately only give this a small share of the score.
        latency_component = 0.5 * features.latency_sensitivity

        # Cost sensitivity pulls the score downward so that callers who want to
        # conserve spend are automatically biased toward cheaper tiers even for
        # tasks that would otherwise be borderline. This avoids a separate
        # post-hoc step that would need to replicate the same thresholds.
        cost_component = 0.2 * features.cost_sensitivity

        raw = (
            0.5 * token_component
            + 0.3 * quality_component
            + 0.2 * latency_component
            - cost_component
        )
        return max(0.0, min(raw, 1.0))

    def _target_tier(self, complexity: float) -> ModelTier:
        if complexity < 0.3:
            return ModelTier.FLASH
        if complexity < 0.7:
            return ModelTier.STANDARD
        return ModelTier.PREMIUM

    def _tier_index(self, tier: ModelTier) -> int:
        """Return the priority index of a tier (lower is more capable)."""

        try:
            return self._tier_order.index(tier)
        except ValueError:
            # Treat LOCAL as strictly below all hosted tiers in priority.
            return len(self._tier_order)

    def _candidate_models(
        self,
        *,
        minimum_tier: ModelTier,
        prefer_tier: ModelTier,
    ) -> Iterable[str]:
        """Return models ordered from highest to lowest tier near the target.

        We iterate tiers starting from the preferred one and only fall back to
        cheaper tiers when higher ones are ruled out by health or budget,
        which keeps behaviour intuitive for callers.
        """

        prefer_idx = self._tier_index(prefer_tier)
        min_idx = self._tier_index(minimum_tier)
        if min_idx < prefer_idx:
            min_idx = prefer_idx

        ordered_tiers = self._tier_order[prefer_idx : min_idx + 1]

        models: list[str] = []
        for tier in ordered_tiers:
            tier_models = sorted(
                m for m, t in self._tiers_by_model.items() if t is tier
            )
            models.extend(tier_models)

        return models

    async def choose_model(self, features: TaskFeatures) -> str:
        """Select the most appropriate model given task features and budgets.

        The method is async to mirror the rest of the pipeline surface, even
        though the current implementation is CPU‑only. This keeps call‑sites
        uniform and leaves room for future data‑driven routing decisions.
        """

        complexity = self._complexity_score(features)
        prefer_tier = self._target_tier(complexity)
        # Ensure we never pick a tier "below" the caller's minimum.
        if self._tier_index(prefer_tier) > self._tier_index(features.minimum_tier):
            prefer_tier = features.minimum_tier

        logger.debug(
            "routing_decision_start",
            complexity=complexity,
            prefer_tier=prefer_tier.value,
            minimum_tier=features.minimum_tier.value,
        )

        candidates = list(
            self._candidate_models(
                minimum_tier=features.minimum_tier,
                prefer_tier=prefer_tier,
            )
        )

        if features.allow_local_fallback:
            # Append local models as absolute last‑chance options; we do not
            # assume they share the same pricing as hosted models.
            local_candidates = [
                m for m, t in self._tiers_by_model.items() if t is ModelTier.LOCAL
            ]
            for m in local_candidates:
                if m not in candidates:
                    candidates.append(m)

        if not candidates:
            raise NoAvailableModelError("No models registered in router")

        for model in candidates:
            breaker = self._breakers_by_model.get(model)
            if breaker is not None and not breaker.can_execute():
                logger.info(
                    "candidate_skipped_circuit_open",
                    model=model,
                    state=breaker.state.name,
                )
                continue

            try:
                if not self._cost_tracker.can_afford(
                    model,
                    input_tokens=features.estimated_input_tokens,
                    output_tokens=features.estimated_output_tokens,
                ):
                    logger.info(
                        "candidate_skipped_budget",
                        model=model,
                    )
                    continue
                # We still compute a cost estimate here for logging even
                # though routing is driven primarily by tier order.
                estimated_cost = self._cost_tracker.estimate_cost(
                    model,
                    input_tokens=features.estimated_input_tokens,
                    output_tokens=features.estimated_output_tokens,
                )
            except KeyError:
                # If we lack pricing for a model we conservatively skip it
                # rather than risk unbounded spend.
                logger.warning(
                    "candidate_skipped_missing_pricing",
                    model=model,
                )
                continue

            logger.info(
                "routing_decision_chosen",
                model=model,
                estimated_cost=str(estimated_cost),
                complexity=complexity,
            )
            return model

        raise NoAvailableModelError("All candidate models rejected by health/budget")

