"""
Token-bucket rate limiter for LLM provider calls.

Centralising throttling here means every caller respects documented provider
limits without reimplementing the logic. The adaptive halving on 429 responses
prevents thundering-herd recovery attempts from compounding quota violations.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog


logger = structlog.get_logger(__name__)

# Default per-model limits sourced from provider documentation.
# rpm = requests per minute, tpm = tokens per minute.
_DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "gemini-2.0-flash": {"rpm": 1500, "tpm": 1_000_000},
    "gpt-4o-mini": {"rpm": 500, "tpm": 200_000},
}


@dataclass
class _Bucket:
    """Mutable token-bucket state for one model.

    Tracking both requests and tokens lets the limiter respect whichever
    constraint would be hit first — many small calls can exhaust RPM before
    TPM, while a few large ones do the reverse.
    """

    tokens_capacity: float
    tokens_refill_per_sec: float
    requests_capacity: float
    requests_refill_per_sec: float
    tokens: float
    requests: float
    last_refill: float = field(default_factory=time.monotonic)
    # Set by on_429; refill rate is halved while now < throttle_until.
    throttle_until: float = 0.0

    def refill(self, now: float) -> None:
        """Replenish buckets using elapsed time, respecting active throttle."""
        elapsed = max(0.0, now - self.last_refill)
        if elapsed <= 0.0:
            return
        # Halve the effective refill rate during throttle window so recovery
        # is gradual rather than immediately returning to full throughput.
        factor = 0.5 if now < self.throttle_until else 1.0
        self.tokens = min(
            self.tokens_capacity,
            self.tokens + elapsed * self.tokens_refill_per_sec * factor,
        )
        self.requests = min(
            self.requests_capacity,
            self.requests + elapsed * self.requests_refill_per_sec * factor,
        )
        self.last_refill = now


class TokenBucketRateLimiter:
    """Async-aware per-model rate limiter with 429-triggered adaptive throttle.

    Keeping this in one place makes it easy to audit aggregate throughput and
    tune limits per model without touching calling code. The lock is released
    during sleep so other coroutines can make progress while one waits.
    """

    def __init__(
        self,
        *,
        model_limits: dict[str, dict[str, int]] | None = None,
    ) -> None:
        limits = model_limits if model_limits is not None else _DEFAULT_LIMITS
        self._buckets: dict[str, _Bucket] = {
            model: self._make_bucket(cfg) for model, cfg in limits.items()
        }
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_bucket(cfg: dict[str, int]) -> _Bucket:
        rpm = float(cfg["rpm"])
        tpm = float(cfg["tpm"])
        return _Bucket(
            tokens_capacity=tpm,
            tokens_refill_per_sec=tpm / 60.0,
            requests_capacity=rpm,
            requests_refill_per_sec=rpm / 60.0,
            # Start full so the first burst of requests isn't rate-limited.
            tokens=tpm,
            requests=rpm,
        )

    def _get_or_create(self, model_id: str) -> _Bucket:
        if model_id not in self._buckets:
            # Permissive fallback for unconfigured models; avoids hard failures
            # during experimentation while still providing some accounting.
            self._buckets[model_id] = self._make_bucket(
                {"rpm": 3000, "tpm": 120_000}
            )
        return self._buckets[model_id]

    async def wait_for_capacity(
        self, model_id: str, estimated_tokens: int
    ) -> None:
        """Block until this model's bucket has capacity for the request.

        Releasing the lock before sleeping allows other models (or other
        callers for the same model) to make progress rather than serialising
        behind a single long wait.
        """
        while True:
            async with self._lock:
                bucket = self._get_or_create(model_id)
                now = time.monotonic()
                bucket.refill(now)

                if bucket.tokens >= estimated_tokens and bucket.requests >= 1.0:
                    bucket.tokens -= estimated_tokens
                    bucket.requests -= 1.0
                    return

                # Compute how long to wait before retrying under the current
                # throttle factor so the sleep duration stays accurate.
                factor = 0.5 if now < bucket.throttle_until else 1.0
                needed_tokens = max(0.0, estimated_tokens - bucket.tokens)
                needed_reqs = max(0.0, 1.0 - bucket.requests)
                eff_tokens_rate = bucket.tokens_refill_per_sec * factor
                eff_reqs_rate = bucket.requests_refill_per_sec * factor
                wait_tokens = (
                    needed_tokens / eff_tokens_rate if needed_tokens > 0 and eff_tokens_rate > 0 else 0.0
                )
                wait_reqs = (
                    needed_reqs / eff_reqs_rate if needed_reqs > 0 and eff_reqs_rate > 0 else 0.0
                )
                wait_for = max(wait_tokens, wait_reqs, 0.01)

                logger.info(
                    "capacity_wait",
                    model_id=model_id,
                    wait_s=round(wait_for, 3),
                    tokens_available=round(bucket.tokens, 1),
                )

            # Sleep outside the lock so other coroutines aren't blocked.
            await asyncio.sleep(wait_for)

    def on_429(self, model_id: str) -> None:
        """Halve the effective rate limit for 60 seconds after a 429.

        Immediately capping available capacity prevents the coroutines that
        are already past the wait_for_capacity gate from racing to retry;
        without this they would collectively retry at full rate.
        """
        bucket = self._buckets.get(model_id)
        if bucket is None:
            return

        now = time.monotonic()
        was_throttled = now < bucket.throttle_until
        bucket.throttle_until = now + 60.0

        # Cap current available tokens/requests to half-capacity immediately.
        bucket.tokens = min(bucket.tokens, bucket.tokens_capacity / 2.0)
        bucket.requests = min(bucket.requests, bucket.requests_capacity / 2.0)

        if not was_throttled:
            logger.warning(
                "throttle_start",
                model_id=model_id,
                duration_s=60.0,
                tokens_capped_at=bucket.tokens,
            )

    def current_capacity(self, model_id: str) -> dict[str, float]:
        """Return a snapshot of available tokens and request slots."""
        bucket = self._buckets.get(model_id)
        if bucket is None:
            return {"tokens": 0.0, "requests": 0.0}
        return {"tokens": bucket.tokens, "requests": bucket.requests}

    def is_throttled(self, model_id: str) -> bool:
        """Return True if the model is currently in a 429-triggered backoff."""
        bucket = self._buckets.get(model_id)
        if bucket is None:
            return False
        return time.monotonic() < bucket.throttle_until
