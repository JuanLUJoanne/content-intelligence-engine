"""
Batch processing primitives for LLM pipelines.

The goal of this module is to make high‑volume processing runs safe to resume
and easy to reason about by centralising idempotency, checkpointing, and dead
letter handling instead of re‑implementing them ad‑hoc in each caller.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Hashable, Optional, Protocol

import structlog

from .checkpoint import Checkpoint, CheckpointManager

if TYPE_CHECKING:
    from src.gateway.batch import BatchSubmitter
    from src.gateway.security import AuditLogger, InputSanitizer, OutputValidator
    from src.pipeline.graph import ContentPipelineGraph


logger = structlog.get_logger(__name__)


class SupportsId(Protocol):
    """Protocol for records that can be uniquely identified.

    Expressing this as a structural protocol lets the processor work with
    plain dicts, ORM models, or custom DTOs as long as they expose a stable
    identifier used for idempotency and checkpointing.
    """

    @property
    def id(self) -> Hashable:  # pragma: no cover - protocol definition
        ...


DLQHandler = Callable[[SupportsId, BaseException], Awaitable[None]]
ProcessFn = Callable[[SupportsId], Awaitable[None]]


@dataclass
class BatchProcessor:
    """Coordinate idempotent batch execution with checkpointing and DLQ routing.

    This class exists so that the pipeline can scale to millions of records
    while still being restartable and observable; by funnelling all execution
    through here we get a single place to enforce at‑least‑once processing
    semantics and isolate poison messages.
    """

    checkpoint_manager: CheckpointManager
    process_fn: ProcessFn
    dlq_handler: DLQHandler
    max_concurrency: int = 10
    graph: Optional[ContentPipelineGraph] = field(default=None)
    sanitizer: Optional[InputSanitizer] = field(default=None)
    validator: Optional[OutputValidator] = field(default=None)
    audit_logger: Optional[AuditLogger] = field(default=None)
    _seen_ids: set[Hashable] = field(default_factory=set, init=False)

    async def _process_single(self, record: SupportsId) -> None:
        record_id = record.id

        if record_id in self._seen_ids:
            logger.debug("record_skipped_duplicate", record_id=str(record_id))
            return

        # --- Graph-based processing path ---
        if self.graph is not None:
            record_dict: dict[str, Any] = {"id": str(record_id)}
            if hasattr(record, "__dict__"):
                record_dict.update(
                    {k: v for k, v in record.__dict__.items() if not k.startswith("_")}
                )

            # 1. Sanitize input before entering graph
            if self.sanitizer is not None:
                try:
                    self.sanitizer.sanitize(json.dumps(record_dict))
                except Exception as exc:
                    logger.warning("record_injection_blocked", record_id=str(record_id))
                    await self.dlq_handler(record, exc)
                    return

            # 2. Run through LangGraph pipeline
            try:
                state = await self.graph.run(record_dict)
            except Exception as exc:
                logger.exception("graph_processing_failed", record_id=str(record_id))
                await self.dlq_handler(record, exc)
                return

            if state.get("sent_to_dlq"):
                await self.dlq_handler(
                    record,
                    RuntimeError(state.get("error") or "sent_to_dlq"),
                )
                return

            # 3. Validate output after graph completes
            if self.validator is not None and state.get("final_output"):
                output_str = json.dumps(state["final_output"])
                if not self.validator.validate(output_str, ""):
                    logger.warning("output_validation_failed", record_id=str(record_id))
                    await self.dlq_handler(record, ValueError("output validation failed"))
                    return

            # 4. Audit log every processed item
            if self.audit_logger is not None:
                await self.audit_logger.log(
                    request_id=str(uuid.uuid4()),
                    input=str(record_id),
                    output=json.dumps(state.get("final_output") or {}),
                    model=state.get("model_id") or "unknown",
                    cost=0.0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )

            self._seen_ids.add(record_id)
            cp = Checkpoint(last_processed_id=str(record_id), metadata={})
            await self.checkpoint_manager.save(cp)
            return

        # --- Original process_fn path ---
        try:
            await self.process_fn(record)
        except Exception as exc:
            logger.exception(
                "record_processing_failed",
                record_id=str(record_id),
            )
            await self.dlq_handler(record, exc)
            return

        self._seen_ids.add(record_id)

        # Persist checkpoint after *successful* processing so that retries
        # always resume from the last confirmed good record.
        cp = Checkpoint(
            last_processed_id=str(record_id),
            metadata={},
        )
        await self.checkpoint_manager.save(cp)

    async def process_batch(
        self,
        records: Iterable[SupportsId],
        *,
        batch_mode: bool = False,
        batch_submitter: Optional[BatchSubmitter] = None,
    ) -> None:
        """Process a stream of records with bounded concurrency.

        When ``batch_mode=True`` a ``BatchSubmitter`` must be provided.
        All records are collected into a JSONL payload, submitted as a single
        batch job, polled until the job reaches a terminal state, and the
        results are retrieved.  This path skips per-record ``process_fn``
        calls because the batch API handles the LLM work remotely.

        When ``batch_mode=False`` (default) each record is processed
        individually via ``process_fn`` with bounded concurrency.
        """

        if batch_mode:
            if batch_submitter is None:
                raise ValueError("batch_submitter is required when batch_mode=True")

            record_list = list(records)
            items = [{"id": str(r.id)} for r in record_list]
            jsonl = batch_submitter.collect(items)

            batch_id = await batch_submitter.submit_batch(jsonl)
            logger.info("batch_mode_submitted", batch_id=batch_id, record_count=len(record_list))

            # Poll until the batch reaches a terminal state.
            while True:
                status = await batch_submitter.poll_batch(batch_id)
                if status.status in ("completed", "failed"):
                    break
                await asyncio.sleep(0)  # yield to the event loop; real code would sleep longer

            results = await batch_submitter.get_results(batch_id)
            logger.info(
                "batch_mode_complete",
                batch_id=batch_id,
                result_count=len(results),
                status=status.status,
            )
            return

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _guarded(record: SupportsId) -> None:
            async with semaphore:
                await self._process_single(record)

        tasks = [asyncio.create_task(_guarded(r)) for r in records]
        if tasks:
            await asyncio.gather(*tasks)

    async def resume_and_process(
        self,
        records: Iterable[SupportsId],
    ) -> None:
        """Skip already‑processed records based on the last checkpoint.

        This helper keeps the resume semantics in one place so that call‑sites
        don't have to reason about where to resume from on every deployment
        or failure; instead they just pass the full logical stream.
        """

        last_id = await self.checkpoint_manager.resume_from_last_processed()
        seen = False if last_id is None else True

        filtered: list[SupportsId] = []
        for record in records:
            if last_id is None:
                filtered.append(record)
                continue

            if not seen:
                if str(record.id) == str(last_id):
                    seen = True
                continue

            filtered.append(record)

        await self.process_batch(filtered)

