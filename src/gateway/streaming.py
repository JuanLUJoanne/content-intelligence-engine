"""
Streaming helpers for incremental LLM responses.

This module hides the mechanics of streaming tokens and enforcing per‑provider
latency budgets so the rest of the stack can treat "streaming vs. non‑streaming"
as a configuration choice rather than a structural difference.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterable, Optional

import structlog


logger = structlog.get_logger(__name__)


TokenStream = AsyncGenerator[str, None]
StreamFactory = Callable[[], TokenStream]


@dataclass
class StreamingResponseHandler:
    """Wrap streaming providers with latency metrics and fallback.

    The handler keeps concerns like time‑to‑first‑token and provider‑specific
    timeouts in one place, making it easy to swap in faster fallbacks when a
    primary model stalls.
    """

    primary: StreamFactory
    fallback: Optional[StreamFactory] = None
    timeout_seconds: float = 15.0
    provider_name: str = "primary"
    fallback_name: str = "fallback"

    async def stream(self) -> TokenStream:
        """Yield tokens from the primary provider, falling back when needed.

        The returned async generator can be plugged into FastAPI's
        `StreamingResponse` or an SSE adapter without coupling the HTTP layer
        to any particular LLM client.
        """

        first_token_logged = False

        async def _emit(factory: StreamFactory, name: str) -> AsyncGenerator[str, None]:
            nonlocal first_token_logged
            start_time = time.perf_counter()
            try:
                async for chunk in self._with_timeout(factory, name):
                    if not first_token_logged:
                        first_token_logged = True
                        ttfb = time.perf_counter() - start_time
                        logger.info(
                            "stream_time_to_first_token",
                            provider=name,
                            time_to_first_token_ms=int(ttfb * 1000),
                        )
                    yield chunk
            except Exception as exc:
                logger.warning(
                    "stream_provider_failed",
                    provider=name,
                    error=str(exc),
                )
                raise

        async def _generator() -> AsyncGenerator[str, None]:
            # Try primary first.
            try:
                async for token in _emit(self.primary, self.provider_name):
                    yield token
                return
            except Exception:
                if self.fallback is None:
                    return
                logger.info(
                    "stream_falling_back_to_secondary",
                    from_provider=self.provider_name,
                    to_provider=self.fallback_name,
                )
                async for token in _emit(self.fallback, self.fallback_name):
                    yield token

        return _generator()

    async def _with_timeout(self, factory: StreamFactory, name: str) -> TokenStream:
        """Apply a timeout to each underlying streaming call.

        We use `asyncio.wait_for` around the provider's stream to avoid
        indefinitely hanging when a single provider stalls.
        """

        # We cannot wrap the async generator itself in wait_for directly, so
        # we enforce the timeout at the level of each `__anext__` call.
        stream = factory()

        async def _iterator() -> AsyncGenerator[str, None]:
            while True:
                try:
                    token = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=self.timeout_seconds,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "stream_timeout",
                        provider=name,
                        timeout_seconds=self.timeout_seconds,
                    )
                    raise
                yield token

        return _iterator()

