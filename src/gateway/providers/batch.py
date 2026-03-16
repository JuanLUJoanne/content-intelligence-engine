"""
Batch providers for cost‑efficient, high‑throughput LLM workloads.

This module abstracts batch execution behind a small interface so callers can
choose between real‑time and delayed, discounted processing without changing
their business logic. Centralising provider‑specific details here keeps the
rest of the pipeline vendor‑agnostic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import structlog


logger = structlog.get_logger(__name__)


class BatchStatus(str, Enum):
    """Normalised view of batch lifecycle across providers."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchProvider(Protocol):
    """Interface implemented by concrete batch providers.

    The protocol is intentionally minimal so that upstream components can
    swap providers (or fan out across them) without being coupled to
    provider‑specific request/response shapes.
    """

    async def submit_batch(self, items: Iterable[Dict[str, Any]]) -> str:  # pragma: no cover - interface
        ...

    async def poll_batch(self, batch_id: str) -> BatchStatus:  # pragma: no cover - interface
        ...

    async def get_results(self, batch_id: str) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        ...


@dataclass
class OpenAIBatchProvider:
    """Adapter for OpenAI's batch API surface.

    In production this class would handle JSONL file creation, upload, batch
    submission, polling, and result download. Here we focus on the control
    flow and leave the low‑level HTTP wiring to be provided by the host
    application, which avoids hard‑coding credentials or SDK choices.
    """

    client: Any  # expected to expose a batch‑capable interface
    tmp_dir: Path

    async def submit_batch(self, items: Iterable[Dict[str, Any]]) -> str:
        """Create a JSONL file, upload it, and submit a batch job.

        The 50% cost reduction compared to real‑time calls is a property of
        the upstream API; at this layer we merely ensure that callers can
        exploit that pricing mode without re‑implementing orchestration.
        """

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.tmp_dir / "openai_batch.jsonl"

        async def _write() -> None:
            import json

            with jsonl_path.open("w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        await asyncio.to_thread(asyncio.run, _write())  # type: ignore[arg-type]

        # The concrete call shape depends on the client; we treat it as an
        # opaque dependency so this module stays decoupled from SDK versions.
        logger.info("openai_batch_submit_start", path=str(jsonl_path))
        batch_id = await self.client.submit_batch_file(str(jsonl_path))
        logger.info("openai_batch_submit_complete", batch_id=batch_id)
        return batch_id

    async def poll_batch(self, batch_id: str) -> BatchStatus:
        """Poll provider until batch reaches a terminal state."""

        raw_status = await self.client.get_batch_status(batch_id)
        mapping = {
            "queued": BatchStatus.PENDING,
            "in_progress": BatchStatus.RUNNING,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
        }
        status = mapping.get(raw_status, BatchStatus.FAILED)
        logger.debug("openai_batch_status", batch_id=batch_id, status=status.value)
        return status

    async def get_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Download and parse the batch results."""

        logger.info("openai_batch_results_fetch_start", batch_id=batch_id)
        results = await self.client.download_batch_results(batch_id)
        # We assume the client returns a list of dicts; any shape coercion
        # should happen in one place so the rest of the pipeline can rely on
        # a uniform representation.
        return list(results)


@dataclass
class GeminiBatchProvider:
    """Adapter for Gemini / Vertex AI style batch prediction.

    This provider assumes an injected Vertex‑aware client and focuses on
    mapping our neutral interface onto the provider's job lifecycle so that
    upstream code does not have to reason about storage buckets or job names.
    """

    client: Any  # expected to wrap Vertex AI batch prediction
    gcs_bucket: str

    async def submit_batch(self, items: Iterable[Dict[str, Any]]) -> str:
        """Stage input files and create a batch prediction job."""

        logger.info("gemini_batch_submit_start", bucket=self.gcs_bucket)
        job_id = await self.client.submit_batch(items=list(items), bucket=self.gcs_bucket)
        logger.info("gemini_batch_submit_complete", job_id=job_id)
        return job_id

    async def poll_batch(self, batch_id: str) -> BatchStatus:
        """Translate provider job status into a normalised enum."""

        raw_status = await self.client.get_job_status(batch_id)
        mapping = {
            "PENDING": BatchStatus.PENDING,
            "RUNNING": BatchStatus.RUNNING,
            "SUCCEEDED": BatchStatus.COMPLETED,
            "FAILED": BatchStatus.FAILED,
            "CANCELLED": BatchStatus.FAILED,
        }
        status = mapping.get(raw_status, BatchStatus.FAILED)
        logger.debug("gemini_batch_status", batch_id=batch_id, status=status.value)
        return status

    async def get_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Fetch and normalise predictions for a completed batch job."""

        logger.info("gemini_batch_results_fetch_start", batch_id=batch_id)
        results = await self.client.download_job_results(batch_id)
        return list(results)

