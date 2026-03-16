"""
Unit tests for ParallelExecutor.

Async lambdas aren't valid Python, so tasks are expressed as zero-argument
async functions or closures that return factories (CallFactory[T]).
"""

from __future__ import annotations

import asyncio

import pytest

from src.gateway.parallel import ParallelExecutor, ParallelResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(value: str):
    """Factory that returns a successful coroutine."""
    async def _coro() -> str:
        return value
    return _coro


def _fail(message: str):
    """Factory that returns a coroutine raising ValueError."""
    async def _coro() -> str:
        raise ValueError(message)
    return _coro


def _slow(delay: float, value: str = "slow"):
    """Factory that sleeps longer than the default timeout."""
    async def _coro() -> str:
        await asyncio.sleep(delay)
        return value
    return _coro


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_tasks_succeed_returns_all_results() -> None:
    executor = ParallelExecutor()
    tasks = [_ok("a"), _ok("b"), _ok("c")]
    result = await executor.execute_parallel(tasks)

    assert isinstance(result, ParallelResult)
    assert sorted(result.successes) == ["a", "b", "c"]
    assert result.failures == []


@pytest.mark.asyncio
async def test_one_failing_task_returns_partial_results() -> None:
    # 3 tasks: ok, fail, ok → successes=[a, c], failures has 1 entry.
    executor = ParallelExecutor()
    tasks = [_ok("a"), _fail("boom"), _ok("c")]
    result = await executor.execute_parallel(tasks)

    assert len(result.successes) == 2
    assert len(result.failures) == 1
    assert isinstance(result.failures[0], ValueError)
    assert sorted(result.successes) == ["a", "c"]


@pytest.mark.asyncio
async def test_slow_task_times_out_while_others_succeed() -> None:
    # The slow task exceeds the timeout and lands in failures; the two fast
    # tasks complete normally. This verifies per-task isolation.
    executor = ParallelExecutor(timeout_seconds=0.05)
    tasks = [_ok("fast1"), _slow(10.0), _ok("fast2")]
    result = await executor.execute_parallel(tasks)

    assert len(result.successes) == 2
    assert len(result.failures) == 1
    assert isinstance(result.failures[0], asyncio.TimeoutError)
    assert sorted(result.successes) == ["fast1", "fast2"]


@pytest.mark.asyncio
async def test_empty_task_list_returns_empty_result() -> None:
    executor = ParallelExecutor()
    result = await executor.execute_parallel([])

    assert result.successes == []
    assert result.failures == []
    assert result.total_latency >= 0.0


@pytest.mark.asyncio
async def test_total_latency_is_recorded() -> None:
    executor = ParallelExecutor()
    result = await executor.execute_parallel([_ok("x")])
    assert result.total_latency >= 0.0


@pytest.mark.asyncio
async def test_all_tasks_fail_returns_all_failures() -> None:
    executor = ParallelExecutor()
    tasks = [_fail("err1"), _fail("err2")]
    result = await executor.execute_parallel(tasks)

    assert result.successes == []
    assert len(result.failures) == 2
