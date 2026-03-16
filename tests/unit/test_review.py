"""Tests for ReviewStore human-in-the-loop review logic."""

from __future__ import annotations

import json

import pytest

from src.api.review import ReviewItem, ReviewStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path) -> ReviewStore:
    return ReviewStore(golden_set_path=str(tmp_path / "golden_set.json"))


def _add(store: ReviewStore, n: int = 1) -> list[str]:
    ids = []
    for i in range(n):
        ids.append(
            store.add_item(
                record={"id": f"rec-{i}"},
                output={"title": f"Title {i}"},
                confidence=0.9,
                reason="test",
            )
        )
    return ids


# ---------------------------------------------------------------------------
# Pending items
# ---------------------------------------------------------------------------


class TestPendingItems:
    def test_newly_added_item_is_pending(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, 1)
        pending = store.get_pending()
        assert len(pending) == 1
        assert pending[0].status == "pending"

    def test_pending_returns_only_pending_items(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _add(store, 3)
        # Approve the first item — it should no longer appear in pending.
        asyncio_run(store.approve(ids[0]))
        pending = store.get_pending()
        assert len(pending) == 2
        pending_ids = {i.id for i in pending}
        assert ids[0] not in pending_ids

    def test_empty_store_returns_empty_pending(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_pending() == []


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_changes_status(self, tmp_path):
        store = _make_store(tmp_path)
        (item_id,) = _add(store, 1)
        item = await store.approve(item_id)
        assert item.status == "approved"

    @pytest.mark.asyncio
    async def test_approve_adds_to_golden_set(self, tmp_path):
        store = _make_store(tmp_path)
        (item_id,) = _add(store, 1)
        await store.approve(item_id)

        golden_path = tmp_path / "golden_set.json"
        assert golden_path.exists()
        entries = json.loads(golden_path.read_text())
        assert len(entries) == 1
        assert entries[0]["record"] == {"id": "rec-0"}
        assert entries[0]["output"] == {"title": "Title 0"}
        assert "approved_at" in entries[0]

    @pytest.mark.asyncio
    async def test_multiple_approvals_all_in_golden_set(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _add(store, 3)
        for item_id in ids:
            await store.approve(item_id)
        entries = json.loads((tmp_path / "golden_set.json").read_text())
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_approve_unknown_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            await store.approve("nonexistent-id")


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_changes_status(self, tmp_path):
        store = _make_store(tmp_path)
        (item_id,) = _add(store, 1)
        item = await store.reject(item_id, reason="hallucination detected")
        assert item.status == "rejected"

    @pytest.mark.asyncio
    async def test_reject_stores_reason(self, tmp_path):
        store = _make_store(tmp_path)
        (item_id,) = _add(store, 1)
        item = await store.reject(item_id, reason="off-topic output")
        assert item.rejection_reason == "off-topic output"

    @pytest.mark.asyncio
    async def test_reject_does_not_write_golden_set(self, tmp_path):
        store = _make_store(tmp_path)
        (item_id,) = _add(store, 1)
        await store.reject(item_id, reason="bad output")
        assert not (tmp_path / "golden_set.json").exists()

    @pytest.mark.asyncio
    async def test_reject_unknown_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            await store.reject("ghost-id", reason="n/a")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_counts_correct(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _add(store, 4)
        await store.approve(ids[0])
        await store.approve(ids[1])
        await store.reject(ids[2], reason="bad")
        # ids[3] stays pending

        s = store.stats()
        assert s["pending"] == 1
        assert s["approved"] == 2
        assert s["rejected"] == 1

    @pytest.mark.asyncio
    async def test_approval_rate_calculated_correctly(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _add(store, 4)
        await store.approve(ids[0])
        await store.approve(ids[1])
        await store.reject(ids[2], reason="x")
        await store.reject(ids[3], reason="y")

        s = store.stats()
        assert s["approval_rate"] == pytest.approx(0.5)

    def test_stats_zero_rate_with_no_decisions(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, 2)
        s = store.stats()
        assert s["approval_rate"] == 0.0
        assert s["pending"] == 2

    def test_empty_store_stats(self, tmp_path):
        store = _make_store(tmp_path)
        s = store.stats()
        assert s == {"pending": 0, "approved": 0, "rejected": 0, "approval_rate": 0.0}


# ---------------------------------------------------------------------------
# ReviewItem dataclass
# ---------------------------------------------------------------------------


class TestReviewItem:
    def test_add_item_returns_string_id(self, tmp_path):
        store = _make_store(tmp_path)
        item_id = store.add_item({"id": "x"}, {"title": "y"})
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    def test_item_fields_set_correctly(self, tmp_path):
        store = _make_store(tmp_path)
        item_id = store.add_item(
            {"id": "r1"}, {"title": "T"}, confidence=0.75, reason="low"
        )
        item = store._items[item_id]
        assert item.record == {"id": "r1"}
        assert item.output == {"title": "T"}
        assert item.confidence == pytest.approx(0.75)
        assert item.reason == "low"
        assert item.rejection_reason is None


# ---------------------------------------------------------------------------
# Helper — run a coroutine synchronously in a test
# ---------------------------------------------------------------------------


def asyncio_run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)
