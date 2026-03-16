"""
Centralised tracking of model usage and spend.

This module keeps a single source of truth for pricing and budget enforcement
so that routing decisions can trade off quality and latency against money
without re‑implementing cost maths in multiple places. By funnelling every
token accounting call through this tracker we make it easy to instrument,
alert, and retrospect on actual LLM spend per model and per pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, Mapping, MutableMapping, NamedTuple, Tuple

import structlog


logger = structlog.get_logger(__name__)


Money = Decimal


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Normalise cost inputs so all arithmetic uses Decimal.

    Using Decimal rather than float avoids subtle rounding drift that becomes
    painful once you start aggregating millions of requests across models.
    """

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class ModelPricing:
    """Per‑model token pricing expressed per 1K tokens.

    Pricing is injected from configuration so it can be updated without code
    changes when vendors adjust their rates or new models are added.
    """

    input_cost_per_1k_tokens: Money
    output_cost_per_1k_tokens: Money


class ModelCostSummary(NamedTuple):
    """Immutable snapshot of spend for a single model.

    Keeping this as a simple tuple makes it easy to serialise for metrics,
    logging or dashboards without coupling callers to internal structures.
    """

    model: str
    total_cost: Money
    total_input_tokens: int
    total_output_tokens: int


class BudgetExceededError(RuntimeError):
    """Raised when a new call would push spend beyond a configured budget.

    Failing fast here forces calling code to handle budget exhaustion
    explicitly instead of silently burning through more tokens.
    """


class CostTracker:
    """Track token usage and enforce budget limits.

    The tracker is intentionally stateful because production pipelines often
    need to make many routing decisions against a shared budget envelope
    (per‑job, per‑tenant, or per‑process). Centralising that state makes the
    trade‑offs visible and testable.
    """

    def __init__(
        self,
        pricing_by_model: Mapping[str, ModelPricing],
        *,
        total_budget: Money | float | int | str | None = None,
        per_request_budget: Money | float | int | str | None = None,
    ) -> None:
        self._pricing: Dict[str, ModelPricing] = dict(pricing_by_model)
        self._cost_by_model: MutableMapping[str, Money] = {}
        self._tokens_by_model: MutableMapping[str, Tuple[int, int]] = {}

        self._total_budget: Money | None = (
            _to_decimal(total_budget) if total_budget is not None else None
        )
        self._per_request_budget: Money | None = (
            _to_decimal(per_request_budget) if per_request_budget is not None else None
        )

    def estimate_cost(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> Money:
        """Return the predicted cost for a call without mutating state.

        Separating estimation from recording lets the router simulate different
        allocations and pick the cheapest option that still respects budgets.
        """

        pricing = self._pricing.get(model)
        if pricing is None:
            raise KeyError(f"No pricing configured for model '{model}'")

        it = Decimal(input_tokens) / Decimal(1000)
        ot = Decimal(output_tokens) / Decimal(1000)

        cost = (
            it * pricing.input_cost_per_1k_tokens
            + ot * pricing.output_cost_per_1k_tokens
        )
        return cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _current_total_cost(self) -> Money:
        return sum(self._cost_by_model.values(), Decimal("0"))

    def can_afford(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        """Cheap predicate to check budgets before committing to a route.

        This lets the router avoid even attempting calls that would clearly
        overshoot the allowed budget envelope for a job or tenant.
        """

        projected = self.estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if self._per_request_budget is not None and projected > self._per_request_budget:
            return False

        if self._total_budget is not None:
            if self._current_total_cost() + projected > self._total_budget:
                return False

        return True

    def record_usage(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> Money:
        """Record a successful call and return its cost.

        Recording is the only place where budgets are enforced; if a call would
        overshoot the allowed envelope we raise so callers can route to a
        cheaper tier, fall back to cached results, or fail the job explicitly.
        """

        cost = self.estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if self._per_request_budget is not None and cost > self._per_request_budget:
            logger.warning(
                "per_request_budget_exceeded",
                model=model,
                cost=str(cost),
                per_request_budget=str(self._per_request_budget),
            )
            raise BudgetExceededError(
                f"Per-request budget of {self._per_request_budget} exceeded: {cost}"
            )

        if self._total_budget is not None:
            projected_total = self._current_total_cost() + cost
            if projected_total > self._total_budget:
                logger.warning(
                    "total_budget_exceeded",
                    model=model,
                    cost=str(cost),
                    projected_total=str(projected_total),
                    total_budget=str(self._total_budget),
                )
                raise BudgetExceededError(
                    f"Total budget of {self._total_budget} would be exceeded: {projected_total}"
                )

        self._cost_by_model[model] = self._cost_by_model.get(model, Decimal("0")) + cost
        in_tok, out_tok = self._tokens_by_model.get(model, (0, 0))
        self._tokens_by_model[model] = (in_tok + input_tokens, out_tok + output_tokens)

        logger.info(
            "usage_recorded",
            model=model,
            cost=str(cost),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return cost

    def summary_by_model(self) -> Iterable[ModelCostSummary]:
        """Return a stable, model‑keyed view of spend for reporting.

        Exposing a read‑only iterator here makes it trivial to push aggregated
        spend into metrics backends or periodic usage reports without reaching
        into internal state.
        """

        for model, total_cost in sorted(self._cost_by_model.items()):
            input_tokens, output_tokens = self._tokens_by_model.get(model, (0, 0))
            yield ModelCostSummary(
                model=model,
                total_cost=total_cost,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
            )

    @property
    def remaining_budget(self) -> Money | None:
        """Return the remaining total budget if one was configured.

        This is mainly useful for observability and dashboards rather than
        routing, but exposing it here makes budget burn‑down very cheap to
        query.
        """

        if self._total_budget is None:
            return None
        return self._total_budget - self._current_total_cost()

