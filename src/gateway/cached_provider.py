"""
Cache-aware wrapper around any LLMProvider.

Wrapping rather than modifying providers keeps caching orthogonal to
transport — you can add/remove the cache layer without touching GeminiProvider
or OpenAIProvider, and tests can verify caching behaviour independently of
any real API call.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.gateway.cache import CacheBackend, make_cache_key
from src.gateway.providers import LLMProvider


logger = structlog.get_logger(__name__)

# Default TTL matches one hour; long enough to absorb eval reruns, short
# enough that stale content doesn't mask real model changes.
_DEFAULT_TTL_SECONDS = 3600


class CachedProvider:
    """LLMProvider decorator that transparently serves responses from cache.

    Having separate hit/miss counters here (in addition to the backend's
    stats()) lets callers attribute cache effectiveness to a specific
    provider+model pair rather than the shared backend as a whole.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: CacheBackend,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    async def generate(self, prompt: str, model_id: str) -> dict[str, Any]:
        """Return a cached response if available; otherwise call the provider.

        Using the prompt as both prompt_text and input_text in the key keeps
        the cache correct even when callers embed all context in a single
        string (the common pattern in this pipeline).
        """
        key = make_cache_key(model_id, prompt, "")
        cached_str = await self._cache.get(key)

        if cached_str is not None:
            self._hits += 1
            logger.debug("cached_provider_hit", model_id=model_id, key_prefix=key[:8])
            return json.loads(cached_str)

        self._misses += 1
        logger.debug("cached_provider_miss", model_id=model_id, key_prefix=key[:8])
        result = await self._provider.generate(prompt, model_id)
        await self._cache.set(key, json.dumps(result), self._ttl, model_id=model_id)
        return result

    @property
    def hit_count(self) -> int:
        """Number of responses served from cache."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Number of responses that required a live provider call."""
        return self._misses
