"""
End-to-end demo for the content intelligence pipeline.

This script exercises the full path from ingest through routing, provider
dispatch, metadata generation, and schema validation so that changes to any
single component can be sanity-checked quickly on a realistic mini workload.
It uses DummyProvider automatically when API keys are absent, which means it
runs in CI and offline environments with zero configuration.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import structlog

from src.gateway.circuit_breaker import CircuitBreaker
from src.gateway.cost_tracker import CostTracker, ModelPricing
from src.gateway.providers import ProviderFactory
from src.gateway.router import ModelRouter, ModelTier, TaskFeatures
from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)

# Real model IDs used both in the router pricing table and by ProviderFactory.
# Using the same string in both places prevents the subtle "model registered
# under one name, looked up under another" class of routing bugs.
_FLASH_MODEL = "gemini-2.0-flash"
_STANDARD_MODEL = "gpt-4o-mini"


def _build_metadata_prompt(image_url: str, caption: str) -> str:
    """Return a prompt that asks the model for structured product catalog metadata.

    The prompt lives here rather than being inlined so the eval harness can
    swap it out during A/B tests without touching pipeline orchestration.
    """
    valid_categories = "electronics, clothing, home, sports, books"
    valid_conditions = "new, like_new, good, fair, refurbished"
    valid_price_ranges = "budget, mid_range, premium, luxury, unpriced"

    return (
        f"You are analyzing a product listing for catalog metadata tagging.\n\n"
        f"Item URL: {image_url}\n"
        f"Description: {caption}\n\n"
        f"Return ONLY a JSON object with these exact fields:\n"
        f"  title: str (clear product title, max 80 chars)\n"
        f"  description: str (2-3 sentences)\n"
        f"  category: one of [{valid_categories}]\n"
        f"  condition: one of [{valid_conditions}]\n"
        f"  price_range: one of [{valid_price_ranges}]\n"
        f"  tags: list of 3-5 lowercase strings\n"
        f"  language: \"en\"\n"
        f"\nDo not include any text outside the JSON."
    )


def _build_router() -> tuple[ModelRouter, CostTracker]:
    """Construct a router wired to the two primary model tiers."""
    pricing = {
        _FLASH_MODEL: ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.075"),
            output_cost_per_1k_tokens=Decimal("0.30"),
        ),
        _STANDARD_MODEL: ModelPricing(
            input_cost_per_1k_tokens=Decimal("0.15"),
            output_cost_per_1k_tokens=Decimal("0.60"),
        ),
    }
    cost_tracker = CostTracker(
        pricing_by_model=pricing,
        total_budget=Decimal("5.0"),
        per_request_budget=Decimal("0.50"),
    )
    return ModelRouter(cost_tracker=cost_tracker), cost_tracker


async def _setup_router(router: ModelRouter) -> None:
    await router.register_model(
        _FLASH_MODEL,
        ModelTier.FLASH,
        breaker=CircuitBreaker(name=_FLASH_MODEL),
    )
    await router.register_model(
        _STANDARD_MODEL,
        ModelTier.STANDARD,
        breaker=CircuitBreaker(name=_STANDARD_MODEL),
    )


def _sample_items() -> list[tuple[str, str]]:
    """Return 5 sample (item_url, description) pairs for the demo.

    The content is illustrative rather than live so the demo is self-contained
    and doesn't depend on network access beyond the LLM API itself.
    """
    base = "https://example.com/product-"
    return [
        (f"{base}001", "Portable Bluetooth speaker with 20-hour battery life"),
        (f"{base}002", "Men's merino wool crew-neck sweater in heather grey"),
        (f"{base}003", "Stainless steel 10-piece cookware set with glass lids"),
        (f"{base}004", "Lightweight trail running shoes with aggressive grip outsole"),
        (f"{base}005", "Wireless noise-cancelling over-ear headphones with 30-hour battery"),
    ]


def _metadata_from_provider_result(
    content_id: str, result: dict[str, Any]
) -> ContentMetadata:
    """Construct and validate a ContentMetadata from a raw provider dict.

    Keeping this in a dedicated function makes it easy to add a retry loop
    (using SchemaRetryPrompt) later without touching the orchestration logic.
    """
    return ContentMetadata(
        content_id=content_id,
        title=result.get("title", "Untitled"),
        description=result.get("description"),
        category=result["category"],
        condition=result["condition"],
        price_range=result["price_range"],
        tags=result.get("tags", []),
        language=result.get("language", "en"),
    )


async def _process_item(
    image_url: str,
    caption: str,
    router: ModelRouter,
    cost_tracker: CostTracker,
) -> ContentMetadata:
    """Run a single item through the full pipeline: route -> generate -> validate."""
    estimated_input = max(len(caption) // 4, 32)
    estimated_output = 200

    features = TaskFeatures(
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=estimated_output,
        latency_sensitivity=0.4,
        quality_sensitivity=0.6,
        cost_sensitivity=0.6,
    )

    model_id = await router.choose_model(features)
    provider = ProviderFactory.get_provider(model_id)
    prompt = _build_metadata_prompt(image_url, caption)

    result = await provider.generate(prompt, model_id)

    cost_tracker.record_usage(
        model_id,
        input_tokens=estimated_input,
        output_tokens=estimated_output,
    )

    metadata = _metadata_from_provider_result(image_url, result)
    logger.info(
        "item_processed",
        content_id=image_url,
        model=model_id,
        category=metadata.category.value,
        condition=metadata.condition.value,
        price_range=metadata.price_range.value,
    )
    return metadata


async def run_demo() -> None:
    """Drive 5 items through the pipeline and print a cost summary."""
    router, cost_tracker = _build_router()
    await _setup_router(router)

    items = _sample_items()
    logger.info("demo_start", total_items=len(items))

    for image_url, caption in items:
        metadata = await _process_item(image_url, caption, router, cost_tracker)
        print(f"\n[{metadata.content_id}]")
        print(f"  title      : {metadata.title}")
        print(f"  category   : {metadata.category.value}")
        print(f"  condition  : {metadata.condition.value}")
        print(f"  price_range: {metadata.price_range.value}")
        print(f"  tags       : {metadata.tags}")

    print("\n=== Cost Summary ===")
    for s in cost_tracker.summary_by_model():
        print(
            f"  {s.model}: ${s.total_cost}"
            f"  (in={s.total_input_tokens} out={s.total_output_tokens} tokens)"
        )
    remaining = cost_tracker.remaining_budget
    if remaining is not None:
        print(f"  remaining budget: ${remaining}")


if __name__ == "__main__":
    asyncio.run(run_demo())
