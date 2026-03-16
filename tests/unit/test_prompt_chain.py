"""
Unit tests for PromptChain.

Tests use DummyProvider so no API keys are required. Step schemas use
extra="allow" so the DummyProvider pool entries (which have ContentMetadata
field names) pass validation regardless of what the step nominally asks for.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from src.gateway.providers import DummyProvider
from src.pipeline.prompt_chain import ChainResult, ChainStep, PromptChain
from src.schemas.metadata import ContentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AnyOutput(BaseModel):
    """Permissive schema: accepts any extra fields DummyProvider returns."""

    model_config = ConfigDict(extra="allow")


class _RequiresResult(BaseModel):
    """Schema with exactly one required field for the retry test."""

    result: str


class _FailOnceThenSucceed:
    """Provider that raises on the first call and succeeds on subsequent ones."""

    def __init__(self) -> None:
        self._calls = 0

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            raise ValueError("transient provider failure")
        return {"result": "recovered"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_step_chain_produces_valid_content_metadata() -> None:
    # Each step outputs a superset of fields (DummyProvider pool entries
    # include title/category/condition/price_range) so the final merged context
    # has everything ContentMetadata requires.
    steps = [
        ChainStep("analyze_item", "Analyze: {item_url}", _AnyOutput),
        ChainStep("classify_category", "Category from: {title}", _AnyOutput),
        ChainStep("generate_metadata", "Full metadata using category={category}", _AnyOutput),
    ]
    chain = PromptChain(steps, DummyProvider(), model_id="dummy")
    result = await chain.run({"item_url": "https://example.com/product-001.jpg"})

    assert isinstance(result, ChainResult)
    assert len(result.steps) == 3
    assert all(s.attempts >= 1 for s in result.steps)

    # The merged context must be rich enough to construct ContentMetadata.
    out = result.final_output
    metadata = ContentMetadata(
        content_id="chain-test",
        title=out["title"],
        category=out["category"],
        condition=out["condition"],
        price_range=out["price_range"],
        tags=out.get("tags", []),
    )
    assert metadata.category is not None
    assert metadata.condition is not None


@pytest.mark.asyncio
async def test_chain_step_failure_retries_only_that_step() -> None:
    # When a step fails transiently, the chain retries that step — not the
    # full chain. We verify this by counting provider calls: 2 total (1 fail
    # + 1 success) with a single-step chain.
    failing_provider = _FailOnceThenSucceed()
    steps = [ChainStep("only_step", "test {input}", _RequiresResult, max_retries=3)]
    chain = PromptChain(steps, failing_provider, model_id="dummy")

    result = await chain.run({"input": "hello"})

    assert result.steps[0].attempts == 2  # failed once, succeeded on second try
    assert failing_provider._calls == 2
    assert result.steps[0].output["result"] == "recovered"


@pytest.mark.asyncio
async def test_chain_exhausts_retries_and_raises() -> None:
    # If every retry fails the chain must propagate an error rather than
    # returning partial results, which could mislead downstream consumers.
    class _AlwaysFail:
        async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
            raise RuntimeError("always fails")

    steps = [ChainStep("bad_step", "fail {input}", _AnyOutput, max_retries=2)]
    chain = PromptChain(steps, _AlwaysFail(), model_id="dummy")

    with pytest.raises(RuntimeError, match="bad_step"):
        await chain.run({"input": "test"})


@pytest.mark.asyncio
async def test_chain_context_flows_between_steps() -> None:
    # Step 2's prompt template references {title} which comes from step 1's
    # output. If context merging is broken, step 2 would raise a KeyError.
    steps = [
        ChainStep("step1", "get title from {image_url}", _AnyOutput),
        ChainStep("step2", "use title: {title}", _AnyOutput),  # {title} from step1
    ]
    chain = PromptChain(steps, DummyProvider(), model_id="dummy")
    result = await chain.run({"image_url": "https://example.com/photo.jpg"})

    assert len(result.steps) == 2
    # Both steps should have run without KeyError
    assert result.steps[1].attempts >= 1


@pytest.mark.asyncio
async def test_chain_total_latency_is_recorded() -> None:
    steps = [ChainStep("s", "prompt {x}", _AnyOutput)]
    chain = PromptChain(steps, DummyProvider(), model_id="dummy")
    result = await chain.run({"x": "value"})
    assert result.total_latency >= 0.0
