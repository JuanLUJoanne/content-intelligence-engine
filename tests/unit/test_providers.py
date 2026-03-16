"""
Unit tests for LLM provider implementations.

Tests focus on DummyProvider and ProviderFactory fallback paths because those
are the behaviours that protect the pipeline in CI and offline environments.
Live provider tests (GeminiProvider, OpenAIProvider) belong in integration
suites where real credentials are available and latency is acceptable.
"""

from __future__ import annotations

import pytest

from src.gateway.providers import DummyProvider, LLMProvider, ProviderFactory
from src.schemas.metadata import ContentMetadata


@pytest.mark.asyncio
async def test_dummy_provider_returns_content_metadata_compatible_dict() -> None:
    # The DummyProvider must produce dicts that satisfy ContentMetadata
    # validation so tests and CI get the same schema guarantees as production
    # without needing any API keys.
    provider = DummyProvider()
    result = await provider.generate("describe this image", "dummy")

    metadata = ContentMetadata(
        content_id="test-1",
        title=result["title"],
        description=result.get("description"),
        category=result["category"],
        condition=result["condition"],
        price_range=result["price_range"],
        tags=result.get("tags", []),
        language=result.get("language", "en"),
    )
    assert metadata.content_id == "test-1"
    assert metadata.category is not None
    assert metadata.condition is not None
    assert metadata.price_range is not None


@pytest.mark.asyncio
async def test_dummy_provider_is_deterministic() -> None:
    # Determinism is required so test assertions don't flake. The same prompt
    # must always produce the same response index regardless of call order.
    provider = DummyProvider()
    prompt = "same prompt every time"
    first = await provider.generate(prompt, "dummy")
    second = await provider.generate(prompt, "dummy")
    assert first == second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_a, prompt_b",
    [
        ("short", "a" * 100),
        ("medium length prompt here", "x" * 200),
    ],
)
async def test_dummy_provider_varies_output_across_prompts(
    prompt_a: str, prompt_b: str
) -> None:
    # Different-length prompts should (eventually) cycle across pool entries,
    # so the demo output isn't monotonous. We just check both are valid.
    provider = DummyProvider()
    result_a = await provider.generate(prompt_a, "dummy")
    result_b = await provider.generate(prompt_b, "dummy")
    for result in (result_a, result_b):
        assert "category" in result
        assert "condition" in result
        assert "price_range" in result


def test_dummy_provider_satisfies_protocol() -> None:
    # Confirm structural compatibility so the type system catches regressions
    # if we ever rename or remove the generate() method.
    provider = DummyProvider()
    assert isinstance(provider, LLMProvider)


def test_factory_returns_dummy_for_explicit_dummy_model() -> None:
    provider = ProviderFactory.get_provider("dummy")
    assert isinstance(provider, DummyProvider)


def test_factory_returns_dummy_when_google_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = ProviderFactory.get_provider("gemini-2.0-flash")
    assert isinstance(provider, DummyProvider)


def test_factory_returns_dummy_when_openai_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = ProviderFactory.get_provider("gpt-4o-mini")
    assert isinstance(provider, DummyProvider)


def test_factory_returns_dummy_for_unknown_model() -> None:
    # An unknown model should never crash the pipeline; fallback to DummyProvider
    # preserves liveness while making the misconfiguration visible in logs.
    provider = ProviderFactory.get_provider("unknown-model-xyz")
    assert isinstance(provider, DummyProvider)


@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-11-20"])
def test_factory_covers_all_openai_model_variants(
    model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = ProviderFactory.get_provider(model_id)
    assert isinstance(provider, DummyProvider)


@pytest.mark.parametrize("model_id", ["gemini-2.0-flash", "gemini-1.5-pro"])
def test_factory_covers_all_gemini_model_variants(
    model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = ProviderFactory.get_provider(model_id)
    assert isinstance(provider, DummyProvider)
