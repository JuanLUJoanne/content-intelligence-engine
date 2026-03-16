"""Tests for DriftDetector baseline management and drift alerting."""

from __future__ import annotations

import pytest

from src.eval.drift_detector import DriftBaseline, DriftDetector, DriftReport
from src.eval.judge import DimensionScore, EvalDimension, EvaluationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(scores: dict[str, float]) -> EvaluationResult:
    """Build an EvaluationResult with a given per-dimension score mapping."""
    return EvaluationResult(
        scores={
            EvalDimension(dim): DimensionScore(score=val, reasoning="test")
            for dim, val in scores.items()
        }
    )


def _stable_scores(value: float = 0.8) -> dict[str, float]:
    return {dim.value: value for dim in EvalDimension}


# ---------------------------------------------------------------------------
# Baseline save and load
# ---------------------------------------------------------------------------


class TestBaselineSaveAndLoad:
    @pytest.mark.asyncio
    async def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        results = [_make_result(_stable_scores(0.8))]

        baseline = await detector.save_baseline("v1", results)

        assert (tmp_path / "baselines.json").exists()
        assert baseline.prompt_version == "v1"
        assert baseline.sample_size == 1

    @pytest.mark.asyncio
    async def test_save_computes_correct_averages(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        results = [
            _make_result({"factual_accuracy": 0.6, "hallucination": 0.8}),
            _make_result({"factual_accuracy": 0.8, "hallucination": 0.6}),
        ]

        baseline = await detector.save_baseline("v1", results)

        assert baseline.dimension_scores["factual_accuracy"] == pytest.approx(0.7)
        assert baseline.dimension_scores["hallucination"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_load_restores_baseline_on_init(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector_a = DriftDetector(baseline_path=path)
        await detector_a.save_baseline("v2", [_make_result(_stable_scores(0.9))])

        detector_b = DriftDetector(baseline_path=path)

        assert detector_b.baseline is not None
        assert detector_b.baseline.prompt_version == "v2"

    @pytest.mark.asyncio
    async def test_empty_results_raises(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        with pytest.raises(ValueError):
            await detector.save_baseline("v1", [])


# ---------------------------------------------------------------------------
# Drift detection — no drift
# ---------------------------------------------------------------------------


class TestNoDrift:
    @pytest.mark.asyncio
    async def test_identical_scores_no_alert(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        results = [_make_result(_stable_scores(0.8))]
        await detector.save_baseline("v1", results)

        report = await detector.detect_drift(results)

        assert report.alert_triggered is False
        assert report.degraded_dimensions == []

    @pytest.mark.asyncio
    async def test_small_drop_below_threshold_no_alert(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        await detector.save_baseline("v1", [_make_result(_stable_scores(0.8))])

        # Drop of 0.04 < threshold of 0.05 → no alert
        current = [_make_result(_stable_scores(0.76))]
        report = await detector.detect_drift(current, threshold=0.05)

        assert report.alert_triggered is False

    @pytest.mark.asyncio
    async def test_no_baseline_returns_no_alert(self, tmp_path):
        path = str(tmp_path / "missing.json")
        detector = DriftDetector(baseline_path=path)

        report = await detector.detect_drift([_make_result(_stable_scores(0.5))])

        assert report.alert_triggered is False
        assert report.per_dimension_deltas == {}


# ---------------------------------------------------------------------------
# Drift detection — drift triggered
# ---------------------------------------------------------------------------


class TestDriftDetected:
    @pytest.mark.asyncio
    async def test_drop_above_threshold_triggers_alert(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        await detector.save_baseline("v1", [_make_result(_stable_scores(0.8))])

        # Drop of 0.1 > threshold of 0.05 → alert
        current = [_make_result(_stable_scores(0.7))]
        report = await detector.detect_drift(current, threshold=0.05)

        assert report.alert_triggered is True

    @pytest.mark.asyncio
    async def test_alert_lists_correct_degraded_dimensions(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        baseline_scores = {
            "factual_accuracy": 0.9,
            "hallucination": 0.9,
            "relevance": 0.9,
        }
        await detector.save_baseline("v1", [_make_result(baseline_scores)])

        # Only factual_accuracy drops significantly
        current_scores = {
            "factual_accuracy": 0.7,   # drops 0.2 → degraded
            "hallucination": 0.88,     # drops 0.02 → not degraded
            "relevance": 0.91,         # improves → not degraded
        }
        report = await detector.detect_drift(
            [_make_result(current_scores)], threshold=0.05
        )

        assert report.alert_triggered is True
        assert "factual_accuracy" in report.degraded_dimensions
        assert "hallucination" not in report.degraded_dimensions
        assert "relevance" not in report.degraded_dimensions

    @pytest.mark.asyncio
    async def test_deltas_sign_convention(self, tmp_path):
        path = str(tmp_path / "baselines.json")
        detector = DriftDetector(baseline_path=path)
        await detector.save_baseline(
            "v1", [_make_result({"factual_accuracy": 0.8})]
        )

        report = await detector.detect_drift(
            [_make_result({"factual_accuracy": 0.6})], threshold=0.05
        )

        # delta = current - baseline = 0.6 - 0.8 = -0.2
        assert report.per_dimension_deltas["factual_accuracy"] == pytest.approx(-0.2)
