"""
Prompt version registry with rollback support.

This module keeps a persistent record of every registered prompt so that
teams can audit changes, roll back quickly on quality regressions, and
automatically revert when a drift alert fires. All state lives in a small
JSON file so the registry works without any database infrastructure.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from src.eval.drift_detector import DriftReport


logger = structlog.get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_git_hash() -> str:
    """Return the current HEAD commit hash, or an empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


@dataclass
class PromptVersion:
    """Immutable record of a registered prompt snapshot."""

    version_id: str
    prompt_text: str
    git_hash: str
    created_at: str                          # ISO-8601 UTC
    eval_scores: Optional[dict[str, float]] = None  # dimension → mean score


class PromptRegistry:
    """Persist and query prompt versions with rollback support.

    The registry tracks which version is "current" so that routing code
    can always serve the active prompt without knowing about the history.
    Rollback is O(1): it simply updates the current pointer and persists.
    """

    def __init__(self, path: str = "data/prompt_versions.json") -> None:
        self._path = Path(path)
        self._versions: list[PromptVersion] = []
        self._current_version_id: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._current_version_id = data.get("current_version_id")
            self._versions = [PromptVersion(**v) for v in data.get("versions", [])]
        except Exception as exc:
            logger.warning("prompt_registry_load_failed", error=str(exc))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_version_id": self._current_version_id,
            "versions": [dataclasses.asdict(v) for v in self._versions],
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        version_id: str,
        prompt_text: str,
        *,
        eval_scores: Optional[dict[str, float]] = None,
    ) -> PromptVersion:
        """Register a new prompt version and make it the current version.

        If a version with the same ``version_id`` already exists it is
        replaced, so re-registering is idempotent for the same content.
        """
        version = PromptVersion(
            version_id=version_id,
            prompt_text=prompt_text,
            git_hash=_get_git_hash(),
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            eval_scores=eval_scores,
        )
        # Replace any existing entry with the same id to stay idempotent.
        self._versions = [v for v in self._versions if v.version_id != version_id]
        self._versions.append(version)
        self._current_version_id = version_id
        self._save()
        logger.info("prompt_registered", version_id=version_id)
        return version

    def get_current(self) -> Optional[PromptVersion]:
        """Return the currently active version, or None if none is registered."""
        if self._current_version_id is None:
            return None
        return self.get_by_version(self._current_version_id)

    def get_by_version(self, version_id: str) -> Optional[PromptVersion]:
        """Look up a specific version by its identifier."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def rollback_to(self, version_id: str) -> None:
        """Set an existing version as the current one.

        Raises ``ValueError`` if the requested version does not exist so that
        callers cannot silently roll back to a non-existent snapshot.
        """
        if self.get_by_version(version_id) is None:
            raise ValueError(f"Version {version_id!r} not found in registry")
        self._current_version_id = version_id
        self._save()
        logger.info("prompt_rollback", version_id=version_id)

    def list_versions(self) -> list[PromptVersion]:
        """Return all registered versions in registration order."""
        return list(self._versions)

    def auto_rollback(self, drift_report: DriftReport) -> Optional[str]:
        """Roll back to the most recent evaluated version if drift is detected.

        When ``drift_report.alert_triggered`` is True, the registry searches
        backwards through its history for the most recently registered version
        that has ``eval_scores`` set (indicating it passed evaluation), skips
        the current version, and rolls back to it.

        Returns the version_id rolled back to, or None if no candidate exists
        or no alert was triggered.
        """
        if not drift_report.alert_triggered:
            return None

        current = self.get_current()
        current_id = current.version_id if current else None

        # Walk history in reverse (newest first) to find last evaluated version
        # that is not the current one.
        candidate: Optional[PromptVersion] = None
        for v in reversed(self._versions):
            if v.version_id == current_id:
                continue
            if v.eval_scores is not None:
                candidate = v
                break

        if candidate is None:
            logger.warning(
                "auto_rollback_no_candidate",
                current_version=current_id,
            )
            return None

        self.rollback_to(candidate.version_id)
        logger.info(
            "prompt_rollback",
            version_id=candidate.version_id,
            reason="auto_rollback_on_drift",
        )
        return candidate.version_id
