"""Tests for observability: Metrics singleton and LangfuseTracker no-op mode."""

from __future__ import annotations

import time

import pytest

from src.observability.metrics import (
    CACHE_HITS_TOTAL,
    LLM_CALL_LATENCY,
    LLM_CALLS_TOTAL,
    Metrics,
)
from src.observability.langfuse_tracker import GenerationSpan, LangfuseTracker, TraceSpan


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset the singleton before each test so counters start at zero."""
    Metrics().reset()
    yield
    Metrics().reset()


class TestMetricsCounter:
    def test_increment_default(self):
        m = Metrics()
        m.inc(LLM_CALLS_TOTAL)
        m.inc(LLM_CALLS_TOTAL)
        assert m.snapshot()["counters"][LLM_CALLS_TOTAL] == 2

    def test_increment_by_value(self):
        m = Metrics()
        m.inc(CACHE_HITS_TOTAL, value=5)
        assert m.snapshot()["counters"][CACHE_HITS_TOTAL] == 5

    def test_counter_with_labels(self):
        m = Metrics()
        m.inc(LLM_CALLS_TOTAL, labels={"model": "gemini-flash"})
        m.inc(LLM_CALLS_TOTAL, labels={"model": "gemini-flash"})
        m.inc(LLM_CALLS_TOTAL, labels={"model": "gpt-4o-mini"})
        snap = m.snapshot()
        assert snap["counters"][LLM_CALLS_TOTAL] == 3
        by_label = snap["counters_by_label"][LLM_CALLS_TOTAL]
        assert by_label["model=gemini-flash"] == 2
        assert by_label["model=gpt-4o-mini"] == 1


class TestMetricsHistogram:
    def test_observe_and_percentiles(self):
        m = Metrics()
        for v in range(1, 101):
            m.observe(LLM_CALL_LATENCY, float(v))
        snap = m.snapshot()
        h = snap["histograms"][LLM_CALL_LATENCY]
        assert h["count"] == 100
        assert h["p50"] == 51.0
        assert h["p95"] == 96.0
        assert h["p99"] == 100.0

    def test_timer_context_manager(self):
        m = Metrics()
        with m.timer("test_op"):
            time.sleep(0.01)
        snap = m.snapshot()
        assert "test_op" in snap["histograms"]
        assert snap["histograms"]["test_op"]["count"] == 1
        assert snap["histograms"]["test_op"]["p50"] >= 0.005


class TestMetricsSnapshot:
    def test_empty_snapshot(self):
        m = Metrics()
        snap = m.snapshot()
        assert snap == {"counters": {}}

    def test_snapshot_format(self):
        m = Metrics()
        m.inc("a")
        m.observe("b", 1.0)
        snap = m.snapshot()
        assert "counters" in snap
        assert "histograms" in snap
        assert snap["counters"]["a"] == 1
        assert snap["histograms"]["b"]["count"] == 1


class TestMetricsSingleton:
    def test_same_instance(self):
        a = Metrics()
        b = Metrics()
        assert a is b

    def test_reset_clears_data(self):
        m = Metrics()
        m.inc("x", value=10)
        m.reset()
        assert m.snapshot()["counters"].get("x", 0) == 0


# ---------------------------------------------------------------------------
# LangfuseTracker
# ---------------------------------------------------------------------------


class TestLangfuseTrackerNoOp:
    def test_disabled_when_no_env(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        tracker = LangfuseTracker()
        assert tracker.enabled is False

    def test_trace_context_manager_yields_span(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracker = LangfuseTracker()
        with tracker.trace("test_trace", metadata={"k": "v"}) as span:
            assert isinstance(span, TraceSpan)
            assert span.name == "test_trace"
            assert span.metadata == {"k": "v"}

    def test_generation_records_in_trace(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracker = LangfuseTracker()
        with tracker.trace("t") as span:
            with span.generation("gen1", model="m", input_data="p") as gen:
                gen.end(output="result", usage={"input_tokens": 10})
        assert len(span.generations) == 1
        assert span.generations[0].output_data == "result"
        assert span.generations[0].usage["input_tokens"] == 10

    def test_generation_records_latency(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracker = LangfuseTracker()
        with tracker.trace("t") as span:
            with span.generation("gen1") as gen:
                time.sleep(0.01)
                gen.end(output="x")
        assert "latency_ms" in span.generations[0].metadata
        assert span.generations[0].metadata["latency_ms"] >= 5.0

    def test_generation_auto_ends_without_explicit_call(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracker = LangfuseTracker()
        with tracker.trace("t") as span:
            with span.generation("gen1") as gen:
                pass  # no explicit gen.end()
        assert "latency_ms" in span.generations[0].metadata

    def test_shutdown_is_safe(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracker = LangfuseTracker()
        tracker.shutdown()  # should not raise
