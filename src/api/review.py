"""
Human-in-the-loop review API.

This module exposes a small FastAPI router for the human review queue so
that low-confidence or flagged pipeline outputs can be inspected, approved,
or rejected before they are used in production or evaluation sets.

Approved items are appended to a golden evaluation set that can be used as
few-shot examples or to measure drift against human judgement.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ReviewItem:
    """A single pipeline output queued for human review."""

    id: str
    record: dict[str, Any]
    output: dict[str, Any]
    confidence: float
    reason: str
    created_at: str   # ISO-8601 UTC
    status: str       # "pending" | "approved" | "rejected"
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# ReviewStore
# ---------------------------------------------------------------------------


class ReviewStore:
    """In-memory store for pending review items with file-backed golden set.

    The golden set is a JSON array written to disk each time an item is
    approved so it can be reused across process restarts for eval baselines
    and prompt improvements.
    """

    def __init__(self, golden_set_path: str = "data/golden_set.json") -> None:
        self._items: dict[str, ReviewItem] = {}
        self._golden_path = Path(golden_set_path)

    # ------------------------------------------------------------------
    # Item management
    # ------------------------------------------------------------------

    def add_item(
        self,
        record: dict[str, Any],
        output: dict[str, Any],
        *,
        confidence: float = 1.0,
        reason: str = "",
    ) -> str:
        """Queue a new item for review and return its assigned id."""
        item_id = str(uuid.uuid4())
        self._items[item_id] = ReviewItem(
            id=item_id,
            record=record,
            output=output,
            confidence=confidence,
            reason=reason,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status="pending",
        )
        return item_id

    def get_pending(self) -> list[ReviewItem]:
        """Return all items with status='pending' in insertion order."""
        return [i for i in self._items.values() if i.status == "pending"]

    async def approve(self, item_id: str) -> ReviewItem:
        """Mark an item as approved and append it to the golden set."""
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"Review item {item_id!r} not found")
        item.status = "approved"
        await self._append_golden(item)
        logger.info("review_approved", item_id=item_id, confidence=item.confidence)
        return item

    async def reject(self, item_id: str, reason: str) -> ReviewItem:
        """Mark an item as rejected and store the rejection reason."""
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"Review item {item_id!r} not found")
        item.status = "rejected"
        item.rejection_reason = reason
        logger.info("review_rejected", item_id=item_id, reason=reason)
        return item

    def stats(self) -> dict[str, Any]:
        """Return a summary of item counts and the approval rate."""
        items = list(self._items.values())
        pending = sum(1 for i in items if i.status == "pending")
        approved = sum(1 for i in items if i.status == "approved")
        rejected = sum(1 for i in items if i.status == "rejected")
        decided = approved + rejected
        return {
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / decided if decided > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Golden set persistence
    # ------------------------------------------------------------------

    async def _append_golden(self, item: ReviewItem) -> None:
        entry = {
            "record": item.record,
            "output": item.output,
            "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        await asyncio.to_thread(self._write_golden_entry, entry)

    def _write_golden_entry(self, entry: dict[str, Any]) -> None:
        self._golden_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if self._golden_path.exists():
            try:
                existing = json.loads(self._golden_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        self._golden_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/review", tags=["review"])

# Module-level singleton used by the router endpoints; callers running tests
# against the store directly should instantiate ReviewStore independently.
_default_store = ReviewStore()


class RejectBody(BaseModel):
    reason: str


def _to_dict(item: ReviewItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "record": item.record,
        "output": item.output,
        "confidence": item.confidence,
        "reason": item.reason,
        "created_at": item.created_at,
        "status": item.status,
        "rejection_reason": item.rejection_reason,
    }


@router.get("/pending")
async def get_pending() -> list[dict[str, Any]]:
    """Return all items currently awaiting human review."""
    return [_to_dict(i) for i in _default_store.get_pending()]


@router.post("/{item_id}/approve")
async def approve_item(item_id: str) -> dict[str, Any]:
    """Approve a review item and add it to the golden evaluation set."""
    try:
        item = await _default_store.approve(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_dict(item)


@router.post("/{item_id}/reject")
async def reject_item(item_id: str, body: RejectBody) -> dict[str, Any]:
    """Reject a review item and record the reason for prompt improvement."""
    try:
        item = await _default_store.reject(item_id, body.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_dict(item)


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Return counts of pending, approved, and rejected items."""
    return _default_store.stats()
