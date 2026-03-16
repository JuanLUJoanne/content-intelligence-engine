"""
Unit tests for the model router.

These tests emphasise routing decisions around cost, complexity, and circuit
breaker state so that changes to heuristics remain safe and explainable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.gateway.circuit_breaker import CircuitBreaker, CircuitState
from src.gateway.cost_tracker import CostTracker, ModelPricing
from src.gateway.router import (
    ModelRouter,
    ModelTier,
    NoAvailableModelError,
    TaskFeatures,
)


def _make_router() -> ModelRouter:
    pricing = {
        "flash-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.10"),
            output_cost_per_1k_tokens=Decimal("0.10"),
        ),
        "standard-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.20"),
            output_cost_per_1k_tokens=Decimal("0.20"),
        ),
        "premium-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.50"),
            output_cost_per_1k_tokens=Decimal("0.50"),
        ),
    }
    tracker = CostTracker(pricing_by_model=pricing, total_budget=Decimal("100.0"))
    return ModelRouter(cost_tracker=tracker)


@pytest.mark.asyncio
async def test_router_prefers_cheapest_model_for_simple_tasks() -> None:
    router = _make_router()
    await router.register_model("flash-model", ModelTier.FLASH)
    await router.register_model("standard-model", ModelTier.STANDARD)
    await router.register_model("premium-model", ModelTier.PREMIUM)

    features = TaskFeatures(
        estimated_input_tokens=256,
        estimated_output_tokens=256,
        latency_sensitivity=0.2,
        quality_sensitivity=0.2,
        cost_sensitivity=0.8,
    )

    chosen = await router.choose_model(features)
    assert chosen == "flash-model"


@pytest.mark.asyncio
async def test_router_falls_back_when_circuit_open() -> None:
    router = _make_router()

    premium_breaker = CircuitBreaker(name="premium-model")
    premium_breaker.state = CircuitState.OPEN

    await router.register_model("flash-model", ModelTier.FLASH)
    await router.register_model("standard-model", ModelTier.STANDARD)
    await router.register_model(
        "premium-model",
        ModelTier.PREMIUM,
        breaker=premium_breaker,
    )

    features = TaskFeatures(
        estimated_input_tokens=4000,
        estimated_output_tokens=4000,
        latency_sensitivity=0.7,
        quality_sensitivity=0.9,
        cost_sensitivity=0.3,
    )

    chosen = await router.choose_model(features)
    # Premium is preferred by complexity but skipped due to open circuit.
    assert chosen == "standard-model"


@pytest.mark.asyncio
async def test_router_respects_budget_constraints() -> None:
    # Configure a very tight budget so the premium model is unaffordable.
    pricing = {
        "flash-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.05"),
            output_cost_per_1k_tokens=Decimal("0.05"),
        ),
        "premium-model": ModelPricing(
            input_cost_per_1k_tokens=Decimal("1.00"),
            output_cost_per_1k_tokens=Decimal("1.00"),
        ),
    }
    tracker = CostTracker(
        pricing_by_model=pricing,
        total_budget=Decimal("1.00"),
        per_request_budget=Decimal("0.30"),
    )
    router = ModelRouter(cost_tracker=tracker)

    await router.register_model("flash-model", ModelTier.FLASH)
    await router.register_model("premium-model", ModelTier.PREMIUM)

    features = TaskFeatures(
        estimated_input_tokens=2000,
        estimated_output_tokens=2000,
        latency_sensitivity=0.5,
        quality_sensitivity=0.8,
        cost_sensitivity=0.8,
    )

    chosen = await router.choose_model(features)
    # Premium would exceed per-request budget; router should pick flash.
    assert chosen == "flash-model"


@pytest.mark.asyncio
async def test_router_raises_when_no_viable_model() -> None:
    router = _make_router()
    # Register a single model that is permanently open-circuited.
    breaker = CircuitBreaker(name="flash-model", failure_threshold=1)
    breaker.state = CircuitState.OPEN
    await router.register_model("flash-model", ModelTier.FLASH, breaker=breaker)

    features = TaskFeatures(
        estimated_input_tokens=100,
        estimated_output_tokens=100,
        latency_sensitivity=0.1,
        quality_sensitivity=0.1,
        cost_sensitivity=0.9,
    )

    with pytest.raises(NoAvailableModelError):
        await router.choose_model(features)

