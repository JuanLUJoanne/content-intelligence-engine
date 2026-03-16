"""
Unit tests for SQLiteCacheBackend and CachedProvider.

Uses pytest's tmp_path fixture so each test gets an isolated SQLite file;
no shared state between tests.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.gateway.cache import SQLiteCacheBackend, make_cache_key
from src.gateway.cached_provider import CachedProvider
from src.gateway.providers import DummyProvider


# ---------------------------------------------------------------------------
# SQLiteCacheBackend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_then_get_returns_same_value(tmp_path: Path) -> None:
    # Round-trip: whatever we store must come back byte-for-byte identical.
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    await backend.set("k1", "hello world", 3600)
    result = await backend.get("k1")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_expired_entry_returns_none(tmp_path: Path) -> None:
    # An entry with ttl=0 should be considered expired on every get() call
    # after any non-zero wall-clock time has passed.
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    await backend.set("k_exp", "should expire", 0)
    # Yield to the event loop so at least one tick passes, making
    # created_at + 0 < time.time() reliably true.
    await asyncio.sleep(0.01)
    result = await backend.get("k_exp")
    assert result is None


@pytest.mark.asyncio
async def test_stats_counts_hits_and_misses(tmp_path: Path) -> None:
    # stats() must reflect the actual number of cache hits and misses so
    # dashboards can compute a meaningful hit rate.
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")

    await backend.get("missing")  # miss
    await backend.set("present", "value", 3600)
    await backend.get("present")  # hit

    stats = backend.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_delete_removes_entry(tmp_path: Path) -> None:
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    await backend.set("to_delete", "bye", 3600)
    await backend.delete("to_delete")
    assert await backend.get("to_delete") is None


@pytest.mark.asyncio
async def test_upsert_overwrites_existing_entry(tmp_path: Path) -> None:
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    await backend.set("upsert_key", "original", 3600)
    await backend.set("upsert_key", "updated", 3600)
    assert await backend.get("upsert_key") == "updated"


@pytest.mark.asyncio
async def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    assert await backend.get("no_such_key") is None


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


def test_make_cache_key_is_deterministic() -> None:
    k1 = make_cache_key("model-a", "prompt text", "input text")
    k2 = make_cache_key("model-a", "prompt text", "input text")
    assert k1 == k2


def test_make_cache_key_varies_with_model() -> None:
    k1 = make_cache_key("model-a", "prompt", "input")
    k2 = make_cache_key("model-b", "prompt", "input")
    assert k1 != k2


# ---------------------------------------------------------------------------
# CachedProvider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_provider_returns_result_on_miss(tmp_path: Path) -> None:
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    provider = DummyProvider()
    cached = CachedProvider(provider, backend)

    result = await cached.generate("test prompt", "dummy")
    assert "category" in result
    assert cached.miss_count == 1
    assert cached.hit_count == 0


@pytest.mark.asyncio
async def test_cached_provider_serves_hit_on_second_call(tmp_path: Path) -> None:
    # The second call with an identical prompt must come from cache, not the
    # provider — confirmed by hit_count incrementing.
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    provider = DummyProvider()
    cached = CachedProvider(provider, backend)

    first = await cached.generate("repeated prompt", "dummy")
    second = await cached.generate("repeated prompt", "dummy")

    assert first == second
    assert cached.hit_count == 1
    assert cached.miss_count == 1


@pytest.mark.asyncio
async def test_cached_provider_isolates_by_model_id(tmp_path: Path) -> None:
    # The same prompt with different model IDs must produce separate cache
    # entries; conflating them would silently serve wrong model responses.
    backend = SQLiteCacheBackend(db_path=tmp_path / "cache.db")
    provider = DummyProvider()
    cached = CachedProvider(provider, backend)

    await cached.generate("same prompt", "model-a")
    await cached.generate("same prompt", "model-b")

    # Both are misses because their keys differ.
    assert cached.miss_count == 2
    assert cached.hit_count == 0
