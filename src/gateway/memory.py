"""
Persistent memory for human-approved LLM outputs.

When a reviewer approves a model response it becomes ground truth for that
input. Storing it here means future identical inputs bypass the LLM entirely,
saving tokens and ensuring consistency. This is the "learning from feedback"
step in a human-in-the-loop pipeline.

Embedding-based fuzzy recall is a planned upgrade; for now we use exact-hash
matching which handles the common case of re-processing unchanged inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.gateway.cache import SQLiteCacheBackend, make_cache_key


logger = structlog.get_logger(__name__)

# Permanent-ish TTL for human-approved entries; 10 years in seconds.
_MEMORY_TTL: int = 10 * 365 * 24 * 3600

# Namespace prefix keeps memory keys distinct from general cache keys so
# the two stores can share the same SQLite file without collisions.
_MEMORY_NS = "memory"


@dataclass
class _QueryRecord:
    key: str
    timestamp: float


class ResponseMemory:
    """Persistent store of human-approved input→output pairs.

    Using SQLiteCacheBackend as the backing store means memory entries share
    the same TTL-aware lookup path as regular cache hits, so recall is O(1)
    and consistent with the rest of the pipeline.

    TODO: Replace exact-hash recall with embedding-based similarity search
    (e.g. pgvector or FAISS) so semantically equivalent inputs hit the same
    memory entry even when phrased differently.
    """

    def __init__(
        self,
        backend: SQLiteCacheBackend,
        *,
        cost_per_call: float = 0.001,
    ) -> None:
        self._backend = backend
        self._cost_per_call = cost_per_call
        self._total_queries: int = 0
        self._cache_hits: int = 0
        self._unique_keys: set[str] = set()
        self._history: list[_QueryRecord] = []

    async def recall(self, input_text: str) -> str | None:
        """Return the stored approved output for this input, or None if absent.

        Every call is counted so stats() can surface hit rates even for inputs
        that have never been approved by a human.
        """
        key = make_cache_key(_MEMORY_NS, input_text, "")
        self._total_queries += 1
        self._unique_keys.add(key)
        self._history.append(_QueryRecord(key=key, timestamp=time.time()))

        result = await self._backend.get(key)
        if result is not None:
            self._cache_hits += 1
            logger.info("memory_recall_hit", key_prefix=key[:8])
        else:
            logger.info("memory_recall_miss", key_prefix=key[:8])
        return result

    async def learn(self, input_text: str, approved_output: str) -> None:
        """Persist a human-approved output so future calls can skip the LLM.

        The long TTL reflects that human approval is high-signal ground truth
        that should not expire on a typical cache schedule.
        """
        key = make_cache_key(_MEMORY_NS, input_text, "")
        await self._backend.set(key, approved_output, _MEMORY_TTL, model_id="memory")
        logger.info("memory_learned", key_prefix=key[:8])

    def stats(self) -> dict[str, Any]:
        """Return aggregate usage counters for observability dashboards."""
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "unique_queries": len(self._unique_keys),
            "cost_saved_by_cache": round(self._cache_hits * self._cost_per_call, 6),
        }

    @property
    def query_history(self) -> list[_QueryRecord]:
        """Ordered list of all recall attempts with timestamps."""
        return list(self._history)
