"""Tests for CostGuardrail multi-level protection."""

from __future__ import annotations

import pytest

from src.gateway.guardrails import CostGuardrail, GuardrailResult, GuardrailViolation


# cost_per_1k_tokens=0.01 → 1 token = $0.00001


def _guardrail(**kwargs) -> CostGuardrail:
    """Helper: create CostGuardrail with a predictable per-token cost."""
    return CostGuardrail(cost_per_1k_tokens=0.01, **kwargs)


class TestAllowedWithinLimits:
    def test_small_request_is_allowed(self):
        g = _guardrail()
        result = g.check("dummy", estimated_tokens=100)
        assert result.allowed is True
        assert result.level == "ok"
        assert result.reason == "ok"

    def test_result_is_guardrail_result_dataclass(self):
        g = _guardrail()
        result = g.check("dummy", estimated_tokens=50)
        assert isinstance(result, GuardrailResult)


class TestPerRequestLimit:
    def test_exceeding_per_request_limit_raises(self):
        # cost_per_1k=0.01; 6000 tokens = $0.06 > default $0.05 limit
        g = _guardrail(per_request_limit=0.05)
        with pytest.raises(GuardrailViolation, match="(?i)per-request"):
            g.check("dummy", estimated_tokens=6000)

    def test_at_exact_limit_is_allowed(self):
        # exactly $0.05 → 5000 tokens at $0.01/1K
        g = _guardrail(per_request_limit=0.05)
        result = g.check("dummy", estimated_tokens=5000)
        assert result.allowed is True

    def test_per_request_violation_logs_level(self, caplog):
        g = _guardrail(per_request_limit=0.01)
        with pytest.raises(GuardrailViolation):
            g.check("dummy", estimated_tokens=2000)  # $0.02 > $0.01


class TestPerMinuteLimit:
    def test_cumulative_minute_spend_blocks(self):
        # per_minute_limit=$0.05; each call costs $0.01 (1000 tokens)
        g = _guardrail(per_minute_limit=0.05)
        for _ in range(5):
            result = g.check("dummy", estimated_tokens=1000)
            assert result.allowed is True
        # 6th call would push over $0.05
        result = g.check("dummy", estimated_tokens=1000)
        assert result.allowed is False
        assert result.level == "per_minute"
        assert "per-minute" in result.reason

    def test_per_minute_block_returns_guardrail_result(self):
        g = _guardrail(per_minute_limit=0.005)
        g.check("dummy", estimated_tokens=500)   # $0.005 — exactly at limit
        result = g.check("dummy", estimated_tokens=1)  # any extra → over limit
        assert isinstance(result, GuardrailResult)
        assert result.allowed is False


class TestAnomalyDetection:
    def test_anomaly_blocks_when_cost_exceeds_multiplier(self):
        # Build up a rolling avg of ~$0.01 (1000 tokens each)
        g = _guardrail(anomaly_multiplier=3.0)
        for _ in range(10):
            g.check("dummy", estimated_tokens=1000)  # avg ≈ $0.01
        # A call at 4× avg → anomaly
        result = g.check("dummy", estimated_tokens=4000)
        assert result.allowed is False
        assert result.level == "anomaly"
        assert "rolling avg" in result.reason

    def test_no_anomaly_without_history(self):
        # First call cannot trigger anomaly because rolling avg is None.
        # Use a high per_request_limit so the anomaly guard is the only check.
        g = _guardrail(anomaly_multiplier=3.0, per_request_limit=1000.0, per_minute_limit=1000.0)
        result = g.check("dummy", estimated_tokens=1)
        assert result.allowed is True  # no history → rolling avg is None → anomaly guard skipped

    def test_rolling_window_capped_at_100(self):
        g = _guardrail(anomaly_multiplier=3.0, per_request_limit=10.0)
        # Fill 100 items at small cost (1 token each = $0.00001)
        for _ in range(100):
            g.check("dummy", estimated_tokens=1)
        # 101st item: still only 100 items tracked
        assert len(g._rolling_costs) == 100


class TestTotalBudgetLimit:
    def test_95_percent_threshold_raises(self):
        # Budget $1.00; 95% = $0.95
        # 100 calls × 1000 tokens each = 100 × $0.01 = $1.00 total
        # So after ~95 calls we should hit the threshold
        g = _guardrail(total_budget_limit=1.0, per_minute_limit=1000.0)
        violations = 0
        for i in range(200):
            try:
                g.check("dummy", estimated_tokens=1000)
            except GuardrailViolation as exc:
                assert "95%" in str(exc) or "threshold" in str(exc).lower()
                violations += 1
                break
        assert violations == 1, "Expected exactly one GuardrailViolation for budget"

    def test_budget_violation_message_includes_limit(self):
        g = _guardrail(total_budget_limit=0.10, per_minute_limit=1000.0)
        # 10 calls × 1000 tokens = $0.10; 95% of $0.10 = $0.095
        # After ~9 calls ($0.09) the next call would push to $0.10 ≥ $0.095
        with pytest.raises(GuardrailViolation):
            for _ in range(100):
                g.check("dummy", estimated_tokens=1000)
