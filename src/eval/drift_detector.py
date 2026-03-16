"""
Prompt quality drift detection for long-running LLM pipelines.

This module compares evaluation scores against a stored baseline so that
gradual degradation in prompt quality does not go unnoticed. Centralising
drift detection here keeps alert thresholds in one place and makes it easy
to wire automatic rollback into any deployment pipeline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from .judge import EvalDimension, EvaluationResult


logger = structlog.get_logger(__name__)


@dataclass
class DriftBaseline:
    """Snapshot of average per-dimension scores for a specific prompt version."""

    prompt_version: str
    dimension_scores: dict[str, float]  # EvalDimension.value → mean score
    created_at: str                      # ISO-8601 UTC timestamp
    sample_size: int


@dataclass
class DriftReport:
    """Outcome of comparing current eval results against the stored baseline."""

    per_dimension_deltas: dict[str, float]  # dimension → (current - baseline); negative = regression
    alert_triggered: bool
    degraded_dimensions: list[str]          # dimensions that dropped > threshold


class DriftDetector:
    """Compare evaluation results against a saved baseline and raise alerts.

    The detector persists baselines to a JSON file so they survive process
    restarts and can be inspected offline. Once a baseline is saved, every
    subsequent call to ``detect_drift`` compares against it and returns a
    ``DriftReport`` that callers can use to trigger rollbacks or notifications.
    """

    def __init__(self, baseline_path: str = "data/drift_baselines.json") -> None:
        self._path = Path(baseline_path)
        self._baseline: Optional[DriftBaseline] = self._load_sync()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_sync(self) -> Optional[DriftBaseline]:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return DriftBaseline(**data)
        except Exception as exc:
            logger.warning("drift_baseline_load_failed", error=str(exc))
            return None

    @staticmethod
    def _avg_scores(results: list[EvaluationResult]) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for result in results:
            for dim, ds in result.scores.items():
                totals.setdefault(dim.value, []).append(ds.score)
        return {dim: sum(vals) / len(vals) for dim, vals in totals.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_baseline(
        self,
        prompt_version: str,
        eval_results: list[EvaluationResult],
    ) -> DriftBaseline:
        """Compute per-dimension averages and persist them as the new baseline.

        Calling this again overwrites any previous baseline; pair it with
        version identifiers so you can trace which prompt produced each one.
        """
        if not eval_results:
            raise ValueError("Cannot save baseline from empty eval_results")

        avg_scores = self._avg_scores(eval_results)
        baseline = DriftBaseline(
            prompt_version=prompt_version,
            dimension_scores=avg_scores,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sample_size=len(eval_results),
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dataclasses.asdict(baseline), ensure_ascii=False, indent=2)
        await asyncio.to_thread(self._path.write_text, payload, "utf-8")

        self._baseline = baseline
        logger.info(
            "drift_baseline_saved",
            prompt_version=prompt_version,
            sample_size=len(eval_results),
        )
        return baseline

    async def detect_drift(
        self,
        current_results: list[EvaluationResult],
        *,
        threshold: float = 0.05,
    ) -> DriftReport:
        """Compare current scores to the baseline and report any regressions.

        A dimension is considered degraded when its average score drops more
        than ``threshold`` below the baseline value. If any dimension degrades
        the report's ``alert_triggered`` flag is set to True.
        """
        if self._baseline is None:
            logger.warning("drift_check_no_baseline")
            return DriftReport(
                per_dimension_deltas={},
                alert_triggered=False,
                degraded_dimensions=[],
            )

        current_avgs = self._avg_scores(current_results)

        deltas: dict[str, float] = {}
        degraded: list[str] = []

        for dim, baseline_score in self._baseline.dimension_scores.items():
            current_score = current_avgs.get(dim)
            if current_score is None:
                continue
            delta = current_score - baseline_score  # negative → regression
            deltas[dim] = delta
            if delta < -threshold:
                degraded.append(dim)

        alert = len(degraded) > 0
        if alert:
            logger.warning(
                "drift_alert",
                degraded_dimensions=degraded,
                threshold=threshold,
            )
        else:
            logger.info(
                "drift_check_passed",
                dimensions_checked=list(deltas.keys()),
                threshold=threshold,
            )

        return DriftReport(
            per_dimension_deltas=deltas,
            alert_triggered=alert,
            degraded_dimensions=degraded,
        )

    @property
    def baseline(self) -> Optional[DriftBaseline]:
        """Return the in-memory baseline, if one has been loaded or saved."""
        return self._baseline
