"""
Parallel orchestration helpers for LLM calls.

Fanning out multiple LLM requests concurrently (e.g. category + condition + tags in
parallel rather than sequentially) reduces end-to-end latency proportionally
to the number of independent tasks. This module enforces timeouts and partial-
result semantics consistently so call sites don't re-invent the same patterns.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, List, Optional, Sequence, TypeVar

import structlog


logger = structlog.get_logger(__name__)

T = TypeVar("T")

CallFactory = Callable[[], Awaitable[T]]


@dataclass
class TaskResult(Generic[T]):
    """Outcome of a single parallel task — either a value or a captured error."""

    value: Optional[T]
    error: Optional[BaseException]

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ParallelResult(Generic[T]):
    """Aggregate result of execute_parallel.

    Keeping successes and failures separate lets callers decide how to handle
    partial completions rather than forcing an all-or-nothing policy.
    """

    successes: List[T]
    failures: List[BaseException]
    total_latency: float


class ParallelExecutor:
    """Run multiple async call factories concurrently with per-task timeouts.

    Centralising timeout and error handling here means individual fan-out
    sites only express *what* to run, not *how* to run it safely under load.
    """

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._timeout = timeout_seconds

    async def execute_parallel(
        self,
        tasks: List[CallFactory[T]],
    ) -> ParallelResult[T]:
        """Run all tasks concurrently; collect successes and failures.

        A timed-out task counts as a failure so callers can surface partial
        results rather than discarding all work when one slow task stalls.
        """
        logger.info("parallel_start", task_count=len(tasks))
        t0 = time.monotonic()
        raw = await self.run(tasks)
        latency = time.monotonic() - t0

        successes: List[T] = [r.value for r in raw if r.ok]  # type: ignore[misc]
        failures: List[BaseException] = [r.error for r in raw if not r.ok]  # type: ignore[misc]

        logger.info(
            "parallel_complete",
            success_count=len(successes),
            fail_count=len(failures),
            latency=round(latency, 3),
        )
        return ParallelResult(
            successes=successes,
            failures=failures,
            total_latency=latency,
        )

    async def run(
        self,
        calls: Sequence[CallFactory[T]],
    ) -> List[TaskResult[T]]:
        """Low-level runner returning per-task TaskResult objects.

        Prefer execute_parallel for normal use; this method is exposed for
        callers that need fine-grained per-task introspection.
        """

        async def _run_single(idx: int, factory: CallFactory[T]) -> TaskResult[T]:
            try:
                value = await asyncio.wait_for(factory(), timeout=self._timeout)
                return TaskResult(value=value, error=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("parallel_call_failed", index=idx, error=str(exc))
                return TaskResult(value=None, error=exc)

        if not calls:
            return []

        tasks = [asyncio.create_task(_run_single(i, f)) for i, f in enumerate(calls)]
        results = await asyncio.gather(*tasks)
        return list(results)
