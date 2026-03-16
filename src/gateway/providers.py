"""
LLM provider clients for the AI gateway.

This module wraps vendor APIs behind a single Protocol so the rest of the
pipeline never imports google.generativeai or openai directly. That isolation
means we can swap backends, add mock implementations for CI, and test routing
logic without live credentials. All imports of optional vendor SDKs are
deferred to __init__ so the module can be loaded even in environments that
only have a subset of packages installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Protocol, runtime_checkable

import structlog


logger = structlog.get_logger(__name__)

# Exponential backoff delays for rate-limited retries. Three attempts cover
# most transient 429 bursts without burning too much wall-clock time.
_RETRY_BACKOFFS: tuple[float, ...] = (1.0, 2.0, 4.0)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every model backend must satisfy.

    Keeping the contract to a single method makes it trivial to add new
    providers or test doubles without touching routing or pipeline code.
    The return dict uses ContentMetadata field names so callers can
    construct a validated schema instance with a single unpack.
    """

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]: ...


class DummyProvider:
    """Deterministic fake provider used when no API keys are present.

    Having a no-op provider means CI, unit tests, and offline demos all
    exercise the same code paths as production without credentials. Without
    this every test that touches the pipeline would need bespoke mocking,
    creating maintenance drag and divergence from real behaviour.
    """

    # Fixed plausible outputs cycling across all valid enum values so callers
    # can make structural assertions without any randomness.
    _POOL: tuple[dict[str, Any], ...] = (
        {
            "title": "Wireless Bluetooth Speaker",
            "description": "Portable speaker with 20-hour battery life and 360-degree sound.",
            "category": "electronics",
            "condition": "new",
            "price_range": "mid_range",
            "tags": ["bluetooth", "speaker", "portable"],
            "language": "en",
        },
        {
            "title": "Men's Merino Wool Sweater",
            "description": "Lightweight merino wool crew-neck sweater in heather grey.",
            "category": "clothing",
            "condition": "like_new",
            "price_range": "premium",
            "tags": ["merino", "sweater", "wool"],
            "language": "en",
        },
        {
            "title": "Stainless Steel Cookware Set",
            "description": "10-piece tri-ply stainless steel cookware set with glass lids.",
            "category": "home",
            "condition": "new",
            "price_range": "premium",
            "tags": ["cookware", "stainless", "kitchen"],
            "language": "en",
        },
        {
            "title": "Trail Running Shoes",
            "description": "Lightweight trail shoes with aggressive grip outsole.",
            "category": "sports",
            "condition": "good",
            "price_range": "mid_range",
            "tags": ["running", "shoes", "lightweight"],
            "language": "en",
        },
        {
            "title": "Noise-Cancelling Headphones",
            "description": "Over-ear headphones with active noise cancellation and 30-hour battery.",
            "category": "electronics",
            "condition": "refurbished",
            "price_range": "luxury",
            "tags": ["headphones", "noise-cancelling", "wireless"],
            "language": "en",
        },
    )

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
        # Use prompt length as a stable selector so repeated calls with the
        # same prompt return the same result, keeping test assertions simple.
        idx = len(prompt) % len(self._POOL)
        result = self._POOL[idx]
        logger.debug("dummy_provider_generate", model_id=model_id, response_idx=idx)
        return result


class GeminiProvider:
    """Google Gemini backend using the google-generativeai SDK.

    All Gemini-specific API calls are isolated here so quota errors, SDK
    version bumps, and retry policies are handled in one place. Callers
    never see vendor exceptions directly; they receive either a structured
    dict or a re-raised exception that the circuit breaker can count.
    """

    def __init__(self, api_key: str) -> None:
        import google.generativeai as genai  # deferred: optional dependency

        genai.configure(api_key=api_key)
        self._genai = genai

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
        model = self._genai.GenerativeModel(model_id)
        last_exc: BaseException | None = None

        for attempt, backoff in enumerate(_RETRY_BACKOFFS):
            t0 = time.monotonic()
            try:
                response = await model.generate_content_async(prompt)
                latency = time.monotonic() - t0
                usage = response.usage_metadata
                logger.info(
                    "gemini_generate_ok",
                    model_id=model_id,
                    latency_s=round(latency, 3),
                    input_tokens=getattr(usage, "prompt_token_count", 0),
                    output_tokens=getattr(usage, "candidates_token_count", 0),
                )
                return json.loads(response.text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                if is_rate_limit and attempt < len(_RETRY_BACKOFFS) - 1:
                    logger.warning(
                        "gemini_rate_limit_retry",
                        model_id=model_id,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.error("gemini_generate_error", model_id=model_id, error=str(exc))
                raise

        raise RuntimeError("Gemini retries exhausted") from last_exc


class OpenAIProvider:
    """OpenAI backend using the official async client.

    Structuring OpenAI calls identically to GeminiProvider means the router
    can treat them interchangeably; only the SDK call and token-counting
    field names differ between the two implementations.
    """

    def __init__(self, api_key: str) -> None:
        from openai import AsyncOpenAI  # deferred: optional dependency

        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            latency = time.monotonic() - t0
            usage = response.usage
            logger.info(
                "openai_generate_ok",
                model_id=model_id,
                latency_s=round(latency, 3),
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            logger.error("openai_generate_error", model_id=model_id, error=str(exc))
            raise


class ProviderFactory:
    """Map model identifiers to live or stub provider instances.

    Centralising this mapping prevents call-sites from making ad-hoc decisions
    about which SDK to import, and ensures the DummyProvider fallback is applied
    consistently when credentials are absent — critical for reproducible CI.
    """

    _OPENAI_MODELS: frozenset[str] = frozenset({"gpt-4o-mini", "gpt-4o", "gpt-4o-2024-11-20"})

    @classmethod
    def get_provider(cls, model_id: str) -> LLMProvider:
        """Return the best available provider for the given model identifier."""
        if model_id == "dummy":
            return DummyProvider()

        if "gemini" in model_id:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                logger.warning("google_api_key_missing", model_id=model_id, fallback="dummy")
                return DummyProvider()
            return GeminiProvider(api_key=api_key)

        if model_id in cls._OPENAI_MODELS:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("openai_api_key_missing", model_id=model_id, fallback="dummy")
                return DummyProvider()
            return OpenAIProvider(api_key=api_key)

        logger.warning("unknown_model_id", model_id=model_id, fallback="dummy")
        return DummyProvider()
