"""Tests for the runtime self-critique (reflection) node."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from src.observability.metrics import REFLECTION_CORRECTIONS_TOTAL, REFLECTION_TOTAL, Metrics
from src.pipeline.reflection import (
    build_critique_prompt,
    build_reextract_prompt,
    parse_critique,
    reflect,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOOD_EXTRACTION: Dict[str, Any] = {
    "title": "Wireless Headphones",
    "category": "electronics",
    "condition": "new",
    "price_range": "mid_range",
    "tags": ["headphones", "wireless"],
    "language": "en",
}

_INPUT_TEXT = json.dumps({
    "id": "test-1",
    "text": "Wireless noise-cancelling headphones with 30-hour battery",
})


class _FakeProvider:
    """Test double whose generate() returns a canned response."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def generate(self, prompt: str, model_id: str) -> Any:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# parse_critique
# ---------------------------------------------------------------------------


class TestParseCritique:
    def test_approved_dict(self) -> None:
        raw = {"approved": True, "issues": [], "suggestion": ""}
        result = parse_critique(raw)
        assert result.approved is True
        assert result.issues == []

    def test_rejected_dict(self) -> None:
        raw = {
            "approved": False,
            "issues": ["category should be clothing"],
            "suggestion": '{"category": "clothing"}',
        }
        result = parse_critique(raw)
        assert result.approved is False
        assert len(result.issues) == 1

    def test_json_string(self) -> None:
        raw = json.dumps({"approved": True, "issues": [], "suggestion": ""})
        result = parse_critique(raw)
        assert result.approved is True

    def test_malformed_input_fails_open(self) -> None:
        result = parse_critique("not json at all")
        assert result.approved is True
        assert result.issues == []

    def test_missing_approved_field_defaults_true(self) -> None:
        result = parse_critique({"issues": ["something"]})
        assert result.approved is True

    def test_issues_as_string_coerced_to_list(self) -> None:
        result = parse_critique({"approved": False, "issues": "single issue"})
        assert result.issues == ["single issue"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_critique_prompt_contains_input_and_extraction(self) -> None:
        prompt = build_critique_prompt(_INPUT_TEXT, _GOOD_EXTRACTION)
        assert "Wireless noise-cancelling" in prompt
        assert "electronics" in prompt
        assert "FACTUAL ACCURACY" in prompt
        assert "HALLUCINATION" in prompt
        assert "SCHEMA COMPLIANCE" in prompt

    def test_reextract_prompt_contains_issues(self) -> None:
        prompt = build_reextract_prompt(
            _INPUT_TEXT, _GOOD_EXTRACTION, ["category wrong"]
        )
        assert "category wrong" in prompt
        assert "ISSUES" in prompt


# ---------------------------------------------------------------------------
# reflect() — approved path
# ---------------------------------------------------------------------------


class TestReflectApproved:
    @pytest.mark.asyncio
    async def test_approved_passes_through_unchanged(self) -> None:
        provider = _FakeProvider([
            {"approved": True, "issues": [], "suggestion": ""},
        ])
        metrics = Metrics()
        metrics.reset()

        output, critique = await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
            metrics=metrics,
        )

        assert output == _GOOD_EXTRACTION
        assert critique.approved is True
        assert provider.call_count == 1  # only critique call, no re-extract
        snapshot = metrics.snapshot()
        assert snapshot["counters"].get(REFLECTION_TOTAL, 0) >= 1

    @pytest.mark.asyncio
    async def test_no_correction_metric_when_approved(self) -> None:
        provider = _FakeProvider([
            {"approved": True, "issues": [], "suggestion": ""},
        ])
        metrics = Metrics()
        metrics.reset()

        await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
            metrics=metrics,
        )

        snapshot = metrics.snapshot()
        assert snapshot["counters"].get(REFLECTION_CORRECTIONS_TOTAL, 0) == 0


# ---------------------------------------------------------------------------
# reflect() — rejected path triggers re-extraction
# ---------------------------------------------------------------------------


class TestReflectRejected:
    @pytest.mark.asyncio
    async def test_rejected_triggers_reextraction(self) -> None:
        corrected = {**_GOOD_EXTRACTION, "category": "clothing"}
        provider = _FakeProvider([
            # First call: critique says rejected
            {"approved": False, "issues": ["category should be clothing"], "suggestion": ""},
            # Second call: re-extraction returns corrected output
            corrected,
        ])
        metrics = Metrics()
        metrics.reset()

        output, critique = await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
            metrics=metrics,
        )

        assert output == corrected
        assert critique.approved is False
        assert provider.call_count == 2  # critique + re-extract
        snapshot = metrics.snapshot()
        assert snapshot["counters"].get(REFLECTION_CORRECTIONS_TOTAL, 0) >= 1

    @pytest.mark.asyncio
    async def test_reextract_failure_returns_original(self) -> None:
        """If re-extraction fails, the original output is preserved."""
        provider = _FakeProvider([
            {"approved": False, "issues": ["bad tags"], "suggestion": ""},
        ])
        # Make the second call raise
        original_generate = provider.generate

        call_idx = 0

        async def _failing_on_second(prompt: str, model_id: str) -> Any:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:
                raise RuntimeError("LLM down")
            return await original_generate(prompt, model_id)

        provider.generate = _failing_on_second  # type: ignore[assignment]

        output, critique = await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
        )

        assert output == _GOOD_EXTRACTION  # original preserved


# ---------------------------------------------------------------------------
# reflect() — fail-open on critique parse error
# ---------------------------------------------------------------------------


class TestReflectFailOpen:
    @pytest.mark.asyncio
    async def test_unparseable_critique_passes_through(self) -> None:
        provider = _FakeProvider([
            {"random_key": "not a critique at all"},
        ])

        output, critique = await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
        )

        # Missing "approved" key defaults to True (fail-open)
        assert output == _GOOD_EXTRACTION
        assert critique.approved is True

    @pytest.mark.asyncio
    async def test_critique_call_failure_passes_through(self) -> None:
        """If the critique LLM call itself fails, pass through unchanged."""
        provider = _FakeProvider([])

        async def _raise(*a: Any, **kw: Any) -> Any:
            raise RuntimeError("network error")

        provider.generate = _raise  # type: ignore[assignment]

        output, critique = await reflect(
            provider=provider,  # type: ignore[arg-type]
            model_id="dummy",
            input_text=_INPUT_TEXT,
            llm_output=_GOOD_EXTRACTION,
        )

        assert output == _GOOD_EXTRACTION
        assert critique.approved is True


# ---------------------------------------------------------------------------
# Feature flag integration (via graph node)
# ---------------------------------------------------------------------------


class TestReflectionNodeInGraph:
    @pytest.mark.asyncio
    async def test_reflection_disabled_skips_node(self) -> None:
        """When reflection_enabled is off, _node_reflect is a no-op."""
        from src.pipeline.graph import ContentPipelineGraph, _initial_state

        graph = ContentPipelineGraph(model_id="dummy")
        state = _initial_state({"id": "1", "text": "test"})
        state = {**state, "llm_output": _GOOD_EXTRACTION, "model_id": "dummy"}

        # Default: reflection_enabled is False in config
        result = await graph._node_reflect(state)
        assert result["llm_output"] == _GOOD_EXTRACTION

    @pytest.mark.asyncio
    async def test_reflection_enabled_runs_critique(self) -> None:
        """When reflection_enabled is on, the node calls reflect()."""
        from src.feature_flags.registry import get_flag_registry, reset_registry
        from src.pipeline.graph import ContentPipelineGraph, _initial_state

        reset_registry()
        flags = get_flag_registry()
        flags.set_override("reflection_enabled", True)

        try:
            graph = ContentPipelineGraph(model_id="dummy")
            state = _initial_state({"id": "2", "text": "headphones"})
            state = {**state, "llm_output": _GOOD_EXTRACTION, "model_id": "dummy"}

            result = await graph._node_reflect(state)
            # DummyProvider returns a dict — reflection ran, output may differ
            assert result["llm_output"] is not None
        finally:
            flags.clear_override("reflection_enabled")
            reset_registry()

    @pytest.mark.asyncio
    async def test_full_pipeline_with_reflection(self) -> None:
        """Smoke test: full pipeline run with reflection enabled."""
        from src.feature_flags.registry import get_flag_registry, reset_registry
        from src.pipeline.graph import ContentPipelineGraph

        reset_registry()
        flags = get_flag_registry()
        flags.set_override("reflection_enabled", True)

        try:
            graph = ContentPipelineGraph(model_id="dummy")
            state = await graph.run({
                "id": "smoke-test",
                "text": "Vintage leather jacket, barely worn",
            })
            # Pipeline should complete without error
            assert state["error"] is None
            assert state["final_output"] is not None
        finally:
            flags.clear_override("reflection_enabled")
            reset_registry()
