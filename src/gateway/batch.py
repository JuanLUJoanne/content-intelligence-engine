"""
In-memory BatchSubmitter for cost-efficient, high-throughput LLM workloads.

The BatchSubmitter interface mirrors the OpenAI Batch API contract (collect
→ submit → poll → get_results) so that swapping to a live provider requires
only injecting a different implementation without touching call sites.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class BatchStatus:
    """Normalised snapshot of a batch job's lifecycle state."""

    status: str       # "pending" | "running" | "completed" | "failed"
    progress_pct: float  # 0.0 – 100.0


class BatchSubmitter:
    """Mock batch submitter that stores state in memory.

    This implementation mirrors the OpenAI Batch API surface so that switching
    to a real provider is a drop-in replacement.  The mock advances through
    pending → running → completed across successive poll_batch calls, which
    lets integration code exercise the full polling loop without live network
    access.
    """

    def __init__(self) -> None:
        self._batches: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, items: list[dict[str, Any]]) -> str:
        """Serialize a list of dicts to a JSONL-formatted string.

        Callers should pass the returned string directly to submit_batch.
        Each line is a self-contained JSON object, matching the format
        expected by the OpenAI Files API.
        """
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in items)

    async def submit_batch(self, jsonl: str) -> str:
        """Store the batch in memory and return a unique batch_id.

        The JSONL string is parsed back into structured items so that
        get_results can return them without re-reading from disk.
        """
        lines = [line for line in jsonl.splitlines() if line.strip()]
        items = [json.loads(line) for line in lines]

        batch_id = str(uuid.uuid4())
        self._batches[batch_id] = {
            "items": items,
            "status": "pending",
            "submitted_at": time.monotonic(),
            "poll_count": 0,
            "results_fetched": False,
        }

        logger.info("batch_submitted", batch_id=batch_id, item_count=len(items))
        return batch_id

    async def poll_batch(self, batch_id: str) -> BatchStatus:
        """Return the current status for a batch job.

        Mock progression:
          poll 1 → running  (50 %)
          poll 2+ → completed (100 %)

        This simulates a real API that requires at least one in-flight poll
        before reaching a terminal state.
        """
        if batch_id not in self._batches:
            return BatchStatus(status="failed", progress_pct=0.0)

        batch = self._batches[batch_id]
        batch["poll_count"] += 1
        poll_count: int = batch["poll_count"]

        if poll_count == 1:
            batch["status"] = "running"
            result = BatchStatus(status="running", progress_pct=50.0)
        else:
            batch["status"] = "completed"
            result = BatchStatus(status="completed", progress_pct=100.0)

        logger.debug(
            "batch_polled",
            batch_id=batch_id,
            status=result.status,
            progress_pct=result.progress_pct,
        )
        return result

    async def get_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Return the items stored for a completed batch.

        In a real provider this would download and parse the output JSONL
        file from object storage.  The mock simply returns the original
        items to keep tests fast and deterministic.
        """
        if batch_id not in self._batches:
            return []

        batch = self._batches[batch_id]
        items: list[dict[str, Any]] = batch["items"]

        if not batch["results_fetched"]:
            batch["results_fetched"] = True
            logger.info("batch_complete", batch_id=batch_id, result_count=len(items))

        return items
