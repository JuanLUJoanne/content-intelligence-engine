"""Tests for BatchSubmitter and processor batch_mode integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.gateway.batch import BatchStatus, BatchSubmitter
from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.processor import BatchProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Rec:
    """Minimal SupportsId-compatible record for processor tests."""

    id: str
    data: str = ""


async def _noop_process(record: Any) -> None:
    pass


async def _noop_dlq(record: Any, exc: BaseException) -> None:
    pass


# ---------------------------------------------------------------------------
# BatchSubmitter — collect
# ---------------------------------------------------------------------------


class TestCollect:
    def test_returns_jsonl_string(self):
        bs = BatchSubmitter()
        items = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        jsonl = bs.collect(items)
        lines = jsonl.splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == items[0]
        assert json.loads(lines[1]) == items[1]

    def test_empty_list_returns_empty_string(self):
        bs = BatchSubmitter()
        result = bs.collect([])
        assert result == ""

    def test_unicode_preserved(self):
        bs = BatchSubmitter()
        items = [{"text": "héllo wörld 🌍"}]
        jsonl = bs.collect(items)
        assert json.loads(jsonl)["text"] == "héllo wörld 🌍"


# ---------------------------------------------------------------------------
# BatchSubmitter — submit / poll / get_results
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    @pytest.mark.asyncio
    async def test_returns_unique_batch_ids(self):
        bs = BatchSubmitter()
        jsonl = bs.collect([{"id": "1"}])
        id1 = await bs.submit_batch(jsonl)
        id2 = await bs.submit_batch(jsonl)
        assert id1 != id2
        assert len(id1) > 0

    @pytest.mark.asyncio
    async def test_batch_stored_in_memory(self):
        bs = BatchSubmitter()
        jsonl = bs.collect([{"id": "x"}])
        batch_id = await bs.submit_batch(jsonl)
        assert batch_id in bs._batches


class TestPollBatch:
    @pytest.mark.asyncio
    async def test_first_poll_returns_running(self):
        bs = BatchSubmitter()
        batch_id = await bs.submit_batch(bs.collect([{"id": "1"}]))
        status = await bs.poll_batch(batch_id)
        assert status.status == "running"
        assert status.progress_pct == 50.0

    @pytest.mark.asyncio
    async def test_second_poll_returns_completed(self):
        bs = BatchSubmitter()
        batch_id = await bs.submit_batch(bs.collect([{"id": "1"}]))
        await bs.poll_batch(batch_id)
        status = await bs.poll_batch(batch_id)
        assert status.status == "completed"
        assert status.progress_pct == 100.0

    @pytest.mark.asyncio
    async def test_unknown_batch_id_returns_failed(self):
        bs = BatchSubmitter()
        status = await bs.poll_batch("nonexistent")
        assert status.status == "failed"

    @pytest.mark.asyncio
    async def test_batch_status_is_dataclass(self):
        bs = BatchSubmitter()
        batch_id = await bs.submit_batch(bs.collect([{"id": "1"}]))
        status = await bs.poll_batch(batch_id)
        assert isinstance(status, BatchStatus)


class TestGetResults:
    @pytest.mark.asyncio
    async def test_returns_submitted_items(self):
        bs = BatchSubmitter()
        items = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
        batch_id = await bs.submit_batch(bs.collect(items))
        results = await bs.get_results(batch_id)
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

    @pytest.mark.asyncio
    async def test_unknown_batch_returns_empty_list(self):
        bs = BatchSubmitter()
        results = await bs.get_results("ghost")
        assert results == []


# ---------------------------------------------------------------------------
# BatchProcessor — batch_mode integration
# ---------------------------------------------------------------------------


class TestProcessBatchBatchMode:
    @pytest.mark.asyncio
    async def test_batch_mode_submits_and_polls(self, tmp_path):
        """process_batch with batch_mode=True uses BatchSubmitter end-to-end."""
        cm = CheckpointManager(tmp_path / "cp.json")
        processor = BatchProcessor(
            checkpoint_manager=cm,
            process_fn=_noop_process,
            dlq_handler=_noop_dlq,
        )
        bs = BatchSubmitter()
        records = [_Rec(id="r1"), _Rec(id="r2"), _Rec(id="r3")]

        await processor.process_batch(records, batch_mode=True, batch_submitter=bs)

        # One batch must have been created and fully processed
        assert len(bs._batches) == 1
        batch = next(iter(bs._batches.values()))
        assert batch["status"] == "completed"
        assert len(batch["items"]) == 3

    @pytest.mark.asyncio
    async def test_batch_mode_requires_submitter(self, tmp_path):
        cm = CheckpointManager(tmp_path / "cp.json")
        processor = BatchProcessor(
            checkpoint_manager=cm,
            process_fn=_noop_process,
            dlq_handler=_noop_dlq,
        )
        with pytest.raises(ValueError, match="batch_submitter"):
            await processor.process_batch([_Rec(id="r1")], batch_mode=True)

    @pytest.mark.asyncio
    async def test_batch_mode_false_uses_process_fn(self, tmp_path):
        """Default batch_mode=False still calls process_fn per record."""
        called = []

        async def _track(record: Any) -> None:
            called.append(record.id)

        cm = CheckpointManager(tmp_path / "cp.json")
        processor = BatchProcessor(
            checkpoint_manager=cm,
            process_fn=_track,
            dlq_handler=_noop_dlq,
        )
        records = [_Rec(id="a"), _Rec(id="b")]
        await processor.process_batch(records)
        assert sorted(called) == ["a", "b"]
