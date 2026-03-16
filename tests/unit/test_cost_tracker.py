"""
Unit tests for the cost tracking primitives.

These tests focus on the budgeting and aggregation rules because those are the
behaviours most likely to cause surprising spend when misconfigured.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.gateway.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    ModelPricing,
)


def _make_tracker(total_budget: str = "10.0", per_request_budget: str = "5.0") -> CostTracker:
    pricing = {
        "flash-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.10"),
            output_cost_per_1k_tokens=Decimal("0.20"),
        ),
        "premium-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.50"),
            output_cost_per_1k_tokens=Decimal("1.00"),
        ),
    }
    return CostTracker(
        pricing_by_model=pricing,
        total_budget=Decimal(total_budget),
        per_request_budget=Decimal(per_request_budget),
    )


def test_estimate_cost_is_token_based() -> None:
    tracker = _make_tracker()
    cost = tracker.estimate_cost(
        "flash-model",
        input_tokens=500,
        output_tokens=1500,
    )
    # 0.5 * 0.10 + 1.5 * 0.20 = 0.05 + 0.30 = 0.35
    assert cost == Decimal("0.3500")


def test_record_usage_updates_totals_and_summary() -> None:
    tracker = _make_tracker()
    tracker.record_usage("flash-model", input_tokens=1000, output_tokens=0)
    tracker.record_usage("flash-model", input_tokens=0, output_tokens=1000)

    summary = list(tracker.summary_by_model())
    assert len(summary) == 1
    s = summary[0]
    assert s.model == "flash-model"
    assert s.total_input_tokens == 1000
    assert s.total_output_tokens == 1000
    # 1 * 0.10 + 1 * 0.20
    assert s.total_cost == Decimal("0.3000")


def test_per_request_budget_is_enforced() -> None:
    tracker = _make_tracker(per_request_budget="0.10")

    # Cheap call should succeed.
    tracker.record_usage("flash-model", input_tokens=100, output_tokens=100)

    # Expensive call should be rejected for the same model.
    with pytest.raises(BudgetExceededError):
        tracker.record_usage("flash-model", input_tokens=5000, output_tokens=5000)


def test_total_budget_limits_accumulated_spend() -> None:
    tracker = _make_tracker(total_budget="0.30")

    tracker.record_usage("flash-model", input_tokens=500, output_tokens=500)
    # Remaining budget is small; this call would overshoot and should fail.
    with pytest.raises(BudgetExceededError):
        tracker.record_usage("premium-model", input_tokens=1000, output_tokens=0)

    remaining = tracker.remaining_budget
    assert remaining is not None
    assert remaining > Decimal("0")

