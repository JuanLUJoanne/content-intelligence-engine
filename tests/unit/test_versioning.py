"""Tests for PromptRegistry version management and auto-rollback."""

from __future__ import annotations

import pytest

from src.eval.drift_detector import DriftReport
from src.pipeline.versioning import PromptVersion, PromptRegistry


# ---------------------------------------------------------------------------
# Registration and retrieval
# ---------------------------------------------------------------------------


class TestRegisterAndGetCurrent:
    def test_first_registered_version_becomes_current(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First prompt")
        assert registry.get_current().version_id == "v1"

    def test_later_registration_updates_current(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First prompt")
        registry.register("v2", "Second prompt")
        assert registry.get_current().version_id == "v2"

    def test_get_current_none_when_empty(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        assert registry.get_current() is None

    def test_register_returns_prompt_version(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        v = registry.register("v1", "My prompt")
        assert isinstance(v, PromptVersion)
        assert v.version_id == "v1"
        assert v.prompt_text == "My prompt"

    def test_register_with_eval_scores(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        scores = {"factual_accuracy": 0.85, "relevance": 0.90}
        v = registry.register("v1", "prompt", eval_scores=scores)
        assert v.eval_scores == scores

    def test_get_by_version_returns_correct_entry(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First")
        registry.register("v2", "Second")
        assert registry.get_by_version("v1").prompt_text == "First"
        assert registry.get_by_version("v2").prompt_text == "Second"

    def test_get_by_version_returns_none_for_unknown(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        assert registry.get_by_version("vX") is None

    def test_list_versions_returns_all(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "A")
        registry.register("v2", "B")
        registry.register("v3", "C")
        ids = [v.version_id for v in registry.list_versions()]
        assert ids == ["v1", "v2", "v3"]

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "versions.json")
        r1 = PromptRegistry(path=path)
        r1.register("v1", "Saved prompt")

        r2 = PromptRegistry(path=path)
        assert r2.get_current().version_id == "v1"
        assert r2.get_current().prompt_text == "Saved prompt"


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_changes_current_version(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First")
        registry.register("v2", "Second")

        registry.rollback_to("v1")

        assert registry.get_current().version_id == "v1"

    def test_rollback_persists(self, tmp_path):
        path = str(tmp_path / "versions.json")
        r1 = PromptRegistry(path=path)
        r1.register("v1", "First")
        r1.register("v2", "Second")
        r1.rollback_to("v1")

        r2 = PromptRegistry(path=path)
        assert r2.get_current().version_id == "v1"

    def test_rollback_to_unknown_raises(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First")
        with pytest.raises(ValueError, match="not found"):
            registry.rollback_to("v99")

    def test_rollback_does_not_delete_versions(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First")
        registry.register("v2", "Second")
        registry.rollback_to("v1")
        assert len(registry.list_versions()) == 2


# ---------------------------------------------------------------------------
# Auto-rollback on drift alert
# ---------------------------------------------------------------------------


class TestAutoRollback:
    def _drift_alert(self) -> DriftReport:
        return DriftReport(
            per_dimension_deltas={"factual_accuracy": -0.1},
            alert_triggered=True,
            degraded_dimensions=["factual_accuracy"],
        )

    def _no_drift(self) -> DriftReport:
        return DriftReport(
            per_dimension_deltas={},
            alert_triggered=False,
            degraded_dimensions=[],
        )

    def test_auto_rollback_returns_none_when_no_alert(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First", eval_scores={"accuracy": 0.9})
        registry.register("v2", "Second")

        result = registry.auto_rollback(self._no_drift())
        assert result is None
        assert registry.get_current().version_id == "v2"

    def test_auto_rollback_reverts_to_last_evaluated_version(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        # v1 has eval_scores (was evaluated and approved)
        registry.register("v1", "First", eval_scores={"accuracy": 0.9})
        # v2 is current and caused drift
        registry.register("v2", "Second")

        rolled_back = registry.auto_rollback(self._drift_alert())

        assert rolled_back == "v1"
        assert registry.get_current().version_id == "v1"

    def test_auto_rollback_returns_none_when_no_candidate(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        # Neither version has eval_scores
        registry.register("v1", "First")
        registry.register("v2", "Second")

        result = registry.auto_rollback(self._drift_alert())
        assert result is None

    def test_auto_rollback_picks_most_recent_evaluated_candidate(self, tmp_path):
        registry = PromptRegistry(path=str(tmp_path / "versions.json"))
        registry.register("v1", "First", eval_scores={"accuracy": 0.7})
        registry.register("v2", "Second", eval_scores={"accuracy": 0.9})
        registry.register("v3", "Third")  # current, causing drift

        rolled_back = registry.auto_rollback(self._drift_alert())

        # Should pick v2 (most recent with eval_scores before v3)
        assert rolled_back == "v2"
