"""
Unit tests for TokenBucketRateLimiter.

Tests use small synthetic limits so we can drain buckets instantly and observe
blocking behaviour within milliseconds, without relying on wall-clock time for
the happy-path cases.
"""

from __future__ import annotations

import asyncio

import pytest

from src.gateway.rate_limiter import TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_limiter(**overrides: int) -> TokenBucketRateLimiter:
    """Limiter with a small, easily-drainable bucket for testing."""
    limits = {"test-model": {"rpm": 60, "tpm": 100}, **overrides}
    return TokenBucketRateLimiter(model_limits=limits)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capacity_decreases_after_request() -> None:
    # After a successful wait_for_capacity call the bucket should have fewer
    # tokens, proving the limiter is actually accounting for consumed tokens.
    limiter = TokenBucketRateLimiter()
    model = "gemini-2.0-flash"

    before = limiter.current_capacity(model)
    await limiter.wait_for_capacity(model, estimated_tokens=1000)
    after = limiter.current_capacity(model)

    assert after["tokens"] < before["tokens"]
    assert after["tokens"] == before["tokens"] - 1000


@pytest.mark.asyncio
async def test_wait_for_capacity_blocks_when_bucket_empty() -> None:
    # When the bucket is drained, wait_for_capacity must block indefinitely
    # (from the caller's perspective) until tokens refill. We verify this by
    # timing out rather than completing a second request.
    limiter = _tiny_limiter()

    # Drain the token bucket entirely.
    await limiter.wait_for_capacity("test-model", estimated_tokens=100)

    # The bucket is now empty (0 tokens); the next call should block.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            limiter.wait_for_capacity("test-model", estimated_tokens=1),
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_429_halves_available_capacity() -> None:
    # After on_429 the immediately-available token count must be at most half
    # of what it was before, confirming the adaptive throttle is applied.
    limiter = TokenBucketRateLimiter()
    model = "gemini-2.0-flash"

    before = limiter.current_capacity(model)
    limiter.on_429(model)
    after = limiter.current_capacity(model)

    assert after["tokens"] <= before["tokens"] / 2
    assert after["requests"] <= before["requests"] / 2


@pytest.mark.asyncio
async def test_429_sets_throttled_flag() -> None:
    # is_throttled must return True immediately after on_429 and should be
    # False before it has been called (sanity check).
    limiter = TokenBucketRateLimiter()
    model = "gemini-2.0-flash"

    assert limiter.is_throttled(model) is False
    limiter.on_429(model)
    assert limiter.is_throttled(model) is True


@pytest.mark.asyncio
async def test_current_capacity_unknown_model_returns_zeros() -> None:
    # An unconfigured model must not raise; returning zeros is safe because
    # callers that check capacity before making a request will block rather
    # than sending traffic to a model that has no accounting.
    limiter = TokenBucketRateLimiter()
    cap = limiter.current_capacity("nonexistent-model-xyz")
    assert cap == {"tokens": 0.0, "requests": 0.0}


@pytest.mark.asyncio
async def test_is_throttled_unknown_model_returns_false() -> None:
    limiter = TokenBucketRateLimiter()
    assert limiter.is_throttled("nonexistent-model-xyz") is False
