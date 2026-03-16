"""
Checkpointing utilities for long‑running batch pipelines.

This module exists to make large LLM processing runs restartable without
requiring heavyweight orchestration infrastructure. By persisting a tiny
amount of state to disk as JSON we can cheaply resume after crashes, deploys,
or vendor incidents while keeping the implementation framework‑agnostic.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class Checkpoint:
    """Minimal representation of progress through a logical stream.

    Using a small, explicit data structure keeps the on‑disk format stable and
    easy to inspect with standard tooling, which helps immensely when
    debugging batch failures at scale.
    """

    last_processed_id: str
    metadata: dict[str, Any]


class CheckpointManager:
    """Persist and restore checkpoint state for idempotent processing.

    The manager is intentionally file‑backed and async‑friendly so it can be
    dropped into existing pipelines without introducing new infrastructure; it
    focuses on durability and simplicity rather than throughput.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self) -> Optional[Checkpoint]:
        """Return the last stored checkpoint, if any.

        This function hides the details of file absence and partial writes so
        callers can treat "no checkpoint" as a single, well‑defined state.
        """

        async with self._lock:
            if not self._path.exists():
                return None

            try:
                contents = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
                raw = json.loads(contents)
                cp = Checkpoint(
                    last_processed_id=str(raw["last_processed_id"]),
                    metadata=dict(raw.get("metadata", {})),
                )
                logger.info(
                    "checkpoint_loaded",
                    path=str(self._path),
                    last_processed_id=cp.last_processed_id,
                )
                return cp
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "checkpoint_load_failed",
                    path=str(self._path),
                    error=str(exc),
                )
                return None

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a new checkpoint snapshot atomically where possible.

        We write to a temporary file and then rename so that readers either see
        the old checkpoint or the new one, but never a half‑written file. This
        pattern avoids subtle corruption when processes crash mid‑write.
        """

        payload = {
            "last_processed_id": checkpoint.last_processed_id,
            "metadata": checkpoint.metadata,
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")

        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(payload, ensure_ascii=False)
            await asyncio.to_thread(tmp_path.write_text, encoded, encoding="utf-8")
            await asyncio.to_thread(tmp_path.replace, self._path)

        logger.info(
            "checkpoint_saved",
            path=str(self._path),
            last_processed_id=checkpoint.last_processed_id,
        )

    async def resume_from_last_processed(self) -> Optional[str]:
        """Convenience helper for pipelines that operate on monotonic IDs.

        Returning just the identifier keeps call‑sites lightweight while still
        letting advanced users inspect the full `Checkpoint` when needed.
        """

        cp = await self.load()
        return cp.last_processed_id if cp is not None else None

