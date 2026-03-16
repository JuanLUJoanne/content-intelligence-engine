"""
Unit tests for ResponseMemory.

Uses tmp_path-backed SQLiteCacheBackend so each test is fully isolated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gateway.cache import SQLiteCacheBackend
from src.gateway.memory import ResponseMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory(tmp_path: Path, *, cost_per_call: float = 0.01) -> ResponseMemory:
    backend = SQLiteCacheBackend(db_path=tmp_path / "memory.db")
    return ResponseMemory(backend, cost_per_call=cost_per_call)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learn_then_recall_returns_approved_output(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.learn("describe the product", "approved: bluetooth speaker")
    result = await mem.recall("describe the product")
    assert result == "approved: bluetooth speaker"


@pytest.mark.asyncio
async def test_recall_unknown_input_returns_none(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    result = await mem.recall("something never learned")
    assert result is None


@pytest.mark.asyncio
async def test_learn_overwrites_previous_approval(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.learn("input", "first approval")
    await mem.learn("input", "second approval")
    result = await mem.recall("input")
    assert result == "second approval"


@pytest.mark.asyncio
async def test_stats_counts_hits_and_misses_correctly(tmp_path: Path) -> None:
    mem = _memory(tmp_path, cost_per_call=0.01)
    await mem.recall("not stored")  # miss
    await mem.learn("stored", "value")
    await mem.recall("stored")  # hit
    await mem.recall("stored")  # hit

    stats = mem.stats()
    assert stats["total_queries"] == 3
    assert stats["cache_hits"] == 2
    assert stats["unique_queries"] == 2  # "not stored" and "stored"


@pytest.mark.asyncio
async def test_stats_cost_saved_reflects_hits(tmp_path: Path) -> None:
    # cost_saved_by_cache = hits × cost_per_call
    mem = _memory(tmp_path, cost_per_call=0.05)
    await mem.learn("q", "answer")
    await mem.recall("q")  # hit
    await mem.recall("q")  # hit

    stats = mem.stats()
    assert abs(stats["cost_saved_by_cache"] - 0.10) < 1e-9


@pytest.mark.asyncio
async def test_query_history_records_timestamps(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.recall("first")
    await mem.recall("second")

    history = mem.query_history
    assert len(history) == 2
    # Timestamps should be non-decreasing.
    assert history[1].timestamp >= history[0].timestamp


@pytest.mark.asyncio
async def test_unique_queries_counts_distinct_inputs(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.recall("alpha")
    await mem.recall("alpha")  # same key
    await mem.recall("beta")  # different key

    assert mem.stats()["unique_queries"] == 2
