"""
Application metrics for monitoring LLM pipeline health.

Uses simple counters/histograms that can be exported to Prometheus, Datadog, or
logged as JSON.  No external dependency required — production deployment adds
prometheus_client or datadog-api-client as a transport layer.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional


class Metrics:
    """Thread-safe application metrics singleton."""

    _instance: Optional[Metrics] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._counter_labels: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: int = 1, labels: Optional[dict[str, str]] = None):
        """Increment a counter."""
        self._counters[name] += value
        if labels:
            label_key = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            self._counter_labels[name][label_key] += value

    def observe(self, name: str, value: float):
        """Record a histogram observation (e.g., latency)."""
        self._histograms[name].append(value)

    def timer(self, name: str):
        """Context manager to time an operation and record as histogram."""

        class _Timer:
            def __init__(self, metrics, metric_name):
                self._metrics = metrics
                self._name = metric_name
                self._start = None

            def __enter__(self):
                self._start = time.monotonic()
                return self

            def __exit__(self, *args):
                elapsed = time.monotonic() - self._start
                self._metrics.observe(self._name, elapsed)

        return _Timer(self, name)

    def snapshot(self) -> dict:
        """Export all metrics as a JSON-serializable dict."""
        result: dict = {"counters": dict(self._counters)}

        for name, values in self._histograms.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                result.setdefault("histograms", {})[name] = {
                    "count": n,
                    "sum": round(sum(sorted_v), 6),
                    "avg": round(sum(sorted_v) / n, 6),
                    "p50": sorted_v[int(n * 0.5)],
                    "p95": sorted_v[int(n * 0.95)],
                    "p99": sorted_v[int(n * 0.99)],
                }

        if self._counter_labels:
            result["counters_by_label"] = {
                name: dict(labels)
                for name, labels in self._counter_labels.items()
            }

        return result

    def reset(self):
        """Reset all metrics. Useful for testing."""
        self._init()


# Pre-defined metric names (prevents typos across call sites)
LLM_CALLS_TOTAL = "llm_calls_total"
LLM_CALL_LATENCY = "llm_call_latency_seconds"
VALIDATION_RETRIES_TOTAL = "validation_retries_total"
VALIDATION_FAILURES_TOTAL = "validation_failures_total"
DLQ_ROUTES_TOTAL = "dlq_routes_total"
REVIEW_ROUTES_TOTAL = "review_routes_total"
ENGINEERING_ROUTES_TOTAL = "engineering_routes_total"
CACHE_HITS_TOTAL = "cache_hits_total"
CACHE_MISSES_TOTAL = "cache_misses_total"
EVAL_SCORE = "eval_score"
COST_PER_RECORD = "cost_per_record_usd"
