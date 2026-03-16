"""
Async queue utilities for decoupling producers and consumers.

This module provides a small abstraction over `asyncio.Queue` so that batch
pipelines can absorb bursts of incoming work without overwhelming downstream
LLM calls, while still retaining enough structure to swap in SQS or Redis
backends later.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import structlog


logger = structlog.get_logger(__name__)


T = TypeVar("T")


WorkerFn = Callable[[T], Awaitable[None]]


@dataclass
class AsyncQueueProcessor(Generic[T]):
    """Coordinate a bounded worker pool over an async queue.

    The processor exists to keep backpressure decisions and worker lifecycle
    management in one place so producers stay simple and consumers stay
    focused on business logic.
    """

    worker_fn: WorkerFn[T]
    max_queue_size: int = 1000
    max_concurrency: int = 10
    backpressure_threshold: int = 800
    _queue: asyncio.Queue[T] = field(init=False)
    _workers: list[asyncio.Task[None]] = field(default_factory=list, init=False)
    _stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)

    async def start(self) -> None:
        """Spawn worker tasks that will drain the queue until stopped."""

        if self._workers:
            return

        async def _worker(idx: int) -> None:
            while not self._stopped:
                item = await self._queue.get()
                try:
                    await self.worker_fn(item)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "queue_worker_error",
                        worker=idx,
                        error=str(exc),
                    )
                finally:
                    self._queue.task_done()

        for i in range(self.max_concurrency):
            task = asyncio.create_task(_worker(i))
            self._workers.append(task)

    async def stop(self) -> None:
        """Signal workers to stop after draining current items."""

        self._stopped = True
        await self._queue.join()
        for task in self._workers:
            task.cancel()
        self._workers.clear()

    async def batch_enqueue(self, items: Iterable[T]) -> None:
        """Enqueue a batch of records, applying backpressure when needed.

        Backpressure is enacted by briefly sleeping when the queue grows past
        a configurable threshold, which gives consumers space to catch up
        before producers continue to push new work.
        """

        async for item in self._iter_with_backpressure(items):
            await self._queue.put(item)

    async def _iter_with_backpressure(self, items: Iterable[T]):
        for item in items:
            if self._queue.qsize() >= self.backpressure_threshold:
                logger.debug(
                    "queue_backpressure_pause",
                    size=self._queue.qsize(),
                    threshold=self.backpressure_threshold,
                )
                await asyncio.sleep(0.1)
            yield item

