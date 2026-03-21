"""Tests for EngineeringStore and /engineering/* HTTP endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.review import EngineeringStore, engineering_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> EngineeringStore:
    return EngineeringStore()


def _add(store: EngineeringStore, *, prompt_version: str = "v1", n: int = 1) -> list[str]:
    ids = []
    for i in range(n):
        ids.append(
            store.add_record(
                raw_llm_output={"unrelated_key": f"text_{i}"},
                validation_error="no expected schema fields found",
                prompt_version=prompt_version,
            )
        )
    return ids


def _make_client() -> TestClient:
    """Return a TestClient wired to a fresh engineering router + store."""
    from src.api import review as review_module

    # Replace the module-level singleton so each test starts with an empty store.
    fresh_store = EngineeringStore()
    review_module._default_engineering_store = fresh_store

    mini_app = FastAPI()
    mini_app.include_router(engineering_router)
    return TestClient(mini_app)


# ---------------------------------------------------------------------------
# EngineeringStore unit tests
# ---------------------------------------------------------------------------


class TestEngineeringStore:
    def test_add_record_returns_string_id(self):
        store = _make_store()
        item_id = store.add_record(
            raw_llm_output={"key": "value"},
            validation_error="bad shape",
            prompt_version="v1",
        )
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    def test_newly_added_record_is_pending(self):
        store = _make_store()
        _add(store, n=1)
        pending = store.get_pending()
        assert len(pending) == 1
        assert pending[0].status == "pending"
        assert pending[0].retry_count == 0

    def test_get_pending_excludes_requeued(self):
        store = _make_store()
        ids = _add(store, n=3)
        store.requeue(ids[0])
        pending = store.get_pending()
        assert len(pending) == 2
        assert all(r.item_id != ids[0] for r in pending)

    def test_requeue_changes_status(self):
        store = _make_store()
        (item_id,) = _add(store, n=1)
        record = store.requeue(item_id)
        assert record.status == "requeued"

    def test_requeue_unknown_raises(self):
        store = _make_store()
        with pytest.raises(KeyError):
            store.requeue("nonexistent-id")

    def test_stats_groups_by_prompt_version(self):
        store = _make_store()
        _add(store, prompt_version="v1", n=3)
        _add(store, prompt_version="v2", n=2)
        stats = store.stats_by_prompt_version()
        assert stats["v1"] == 3
        assert stats["v2"] == 2

    def test_stats_excludes_requeued_records(self):
        store = _make_store()
        ids = _add(store, prompt_version="v1", n=2)
        store.requeue(ids[0])
        stats = store.stats_by_prompt_version()
        assert stats.get("v1") == 1

    def test_empty_store_returns_empty_pending(self):
        store = _make_store()
        assert store.get_pending() == []

    def test_empty_store_stats_is_empty(self):
        store = _make_store()
        assert store.stats_by_prompt_version() == {}

    def test_record_fields_set_correctly(self):
        store = _make_store()
        item_id = store.add_record(
            raw_llm_output={"x": 1},
            validation_error="missing fields",
            prompt_version="v3",
            retry_count=0,
        )
        record = store._records[item_id]
        assert record.raw_llm_output == {"x": 1}
        assert record.validation_error == "missing fields"
        assert record.prompt_version == "v3"
        assert record.retry_count == 0
        assert record.failed_at  # ISO-8601 string, non-empty


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestEngineeringEndpoints:
    def test_get_pending_returns_structural_failures(self):
        client = _make_client()
        from src.api import review as review_module

        review_module._default_engineering_store.add_record(
            raw_llm_output={"bad": "output"},
            validation_error="no fields",
            prompt_version="v1",
        )

        resp = client.get("/engineering/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["prompt_version"] == "v1"
        assert data[0]["status"] == "pending"
        assert "raw_llm_output" in data[0]
        assert "validation_error" in data[0]

    def test_get_pending_empty_when_no_failures(self):
        client = _make_client()
        resp = client.get("/engineering/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_requeue_removes_from_pending(self):
        client = _make_client()
        from src.api import review as review_module

        item_id = review_module._default_engineering_store.add_record(
            raw_llm_output={"bad": "output"},
            validation_error="no fields",
            prompt_version="v1",
        )

        resp = client.post(f"/engineering/{item_id}/requeue")
        assert resp.status_code == 200
        assert resp.json()["status"] == "requeued"

        # Should no longer appear in pending.
        pending = client.get("/engineering/pending").json()
        assert all(r["item_id"] != item_id for r in pending)

    def test_requeue_unknown_returns_404(self):
        client = _make_client()
        resp = client.post("/engineering/nonexistent-id/requeue")
        assert resp.status_code == 404

    def test_get_stats_groups_by_prompt_version(self):
        client = _make_client()
        from src.api import review as review_module

        for _ in range(3):
            review_module._default_engineering_store.add_record(
                raw_llm_output={},
                validation_error="e",
                prompt_version="v1",
            )
        for _ in range(2):
            review_module._default_engineering_store.add_record(
                raw_llm_output={},
                validation_error="e",
                prompt_version="v2",
            )

        resp = client.get("/engineering/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["v1"] == 3
        assert stats["v2"] == 2

    def test_get_stats_excludes_requeued(self):
        client = _make_client()
        from src.api import review as review_module

        ids = []
        for _ in range(3):
            ids.append(
                review_module._default_engineering_store.add_record(
                    raw_llm_output={},
                    validation_error="e",
                    prompt_version="v1",
                )
            )
        review_module._default_engineering_store.requeue(ids[0])

        resp = client.get("/engineering/stats")
        assert resp.json()["v1"] == 2
