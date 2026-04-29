# P2-1: Add Langfuse LLM Observability Integration (content-intelligence-engine)

## Context
The project has a `src/observability/__init__.py` (empty) and `@traceable` no-op decorators. The resume claims AI observability experience (Langfuse + LangSmith). We need a real Langfuse integration layer — not a mock, but a proper wrapper that traces LLM calls with input/output/cost/latency, with a graceful fallback when Langfuse is not configured.

Key files to read first:
- `src/gateway/providers.py` — LLM provider abstraction (where LLM calls happen)
- `src/pipeline/graph.py` — Pipeline orchestration (where to trace node execution)
- `src/eval/judge.py` — LLM-as-Judge calls (another trace point)
- `src/gateway/cost_tracker.py` — Cost data (attach to traces)

## What to do

### Step 1: Add langfuse dependency
Add `langfuse` to `pyproject.toml` dependencies (or `requirements.txt`, whichever the project uses):
```
langfuse>=2.0.0
```

### Step 2: Create Langfuse wrapper
Create `src/observability/langfuse_tracker.py`:

```python
"""
Langfuse LLM observability integration.

Wraps LLM calls with generation tracking: input, output, model, latency, cost, tokens.
Gracefully degrades to no-op when LANGFUSE_PUBLIC_KEY is not set.

Usage:
    tracker = LangfuseTracker()  # reads from env

    with tracker.trace("process_record", metadata={"record_id": "abc"}) as trace:
        with trace.generation("llm_call", model="gemini-2.0-flash", input=prompt) as gen:
            result = await provider.generate(prompt)
            gen.end(output=result, usage={"input_tokens": X, "output_tokens": Y})

        with trace.generation("judge_eval", model="gpt-4o-mini", input=eval_prompt) as gen:
            score = await judge.evaluate(result)
            gen.end(output=score)
"""
import os
import time
import logging
from contextlib import contextmanager
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GenerationSpan:
    """Tracks a single LLM generation call."""
    name: str
    model: str
    input_data: Any
    start_time: float = field(default_factory=time.monotonic)
    output_data: Any = None
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _langfuse_generation: Any = None  # real Langfuse generation object

    def end(self, output: Any = None, usage: Optional[dict] = None, metadata: Optional[dict] = None):
        self.output_data = output
        if usage:
            self.usage.update(usage)
        if metadata:
            self.metadata.update(metadata)
        latency_ms = (time.monotonic() - self.start_time) * 1000
        self.metadata["latency_ms"] = round(latency_ms, 2)

        if self._langfuse_generation:
            try:
                self._langfuse_generation.end(
                    output=str(output)[:2000] if output else None,
                    usage=self.usage or None,
                    metadata=self.metadata,
                )
            except Exception as e:
                logger.warning("langfuse generation end failed: %s", e)


@dataclass
class TraceSpan:
    """Tracks a full pipeline trace with nested generations."""
    name: str
    metadata: dict = field(default_factory=dict)
    generations: list = field(default_factory=list)
    _langfuse_trace: Any = None  # real Langfuse trace object
    _tracker: Any = None

    @contextmanager
    def generation(self, name: str, model: str = "", input_data: Any = None):
        gen = GenerationSpan(name=name, model=model, input_data=input_data)

        if self._langfuse_trace:
            try:
                gen._langfuse_generation = self._langfuse_trace.generation(
                    name=name,
                    model=model,
                    input=str(input_data)[:5000] if input_data else None,
                )
            except Exception as e:
                logger.warning("langfuse generation start failed: %s", e)

        self.generations.append(gen)
        try:
            yield gen
        finally:
            if gen.output_data is None:
                gen.end()


class LangfuseTracker:
    """
    LLM observability tracker backed by Langfuse.

    Falls back to no-op logging when LANGFUSE_PUBLIC_KEY is not set.
    This means the code works identically in dev (no Langfuse) and prod (with Langfuse).
    """

    def __init__(self):
        self._client = None
        self._enabled = False

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                self._enabled = True
                logger.info("Langfuse tracking enabled (host=%s)", host)
            except ImportError:
                logger.warning("langfuse package not installed; tracking disabled")
            except Exception as e:
                logger.warning("Langfuse init failed: %s; tracking disabled", e)
        else:
            logger.info("Langfuse keys not set; tracking disabled (no-op mode)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def trace(self, name: str, metadata: Optional[dict] = None):
        span = TraceSpan(name=name, metadata=metadata or {}, _tracker=self)

        if self._client:
            try:
                span._langfuse_trace = self._client.trace(
                    name=name,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("langfuse trace start failed: %s", e)

        try:
            yield span
        finally:
            # Flush on trace end
            if self._client:
                try:
                    self._client.flush()
                except Exception as e:
                    logger.warning("langfuse flush failed: %s", e)

    def shutdown(self):
        if self._client:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception:
                pass
```

### Step 3: Create Prometheus-style metrics module
Create `src/observability/metrics.py`:

```python
"""
Application metrics for monitoring LLM pipeline health.

Uses simple counters/histograms that can be exported to Prometheus, Datadog, or logged as JSON.
No external dependency required — production deployment adds prometheus_client or datadog-api-client.
"""
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HistogramBucket:
    le: float  # less-than-or-equal boundary
    count: int = 0


class Metrics:
    """Thread-safe application metrics singleton."""

    _instance: Optional["Metrics"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._counter_labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
        result = {"counters": dict(self._counters)}

        for name, values in self._histograms.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                result.setdefault("histograms", {})[name] = {
                    "count": n,
                    "sum": sum(sorted_v),
                    "avg": sum(sorted_v) / n,
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


# Pre-defined metric names as constants (prevents typos)
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
```

### Step 4: Integrate into pipeline
Edit `src/pipeline/graph.py` to use metrics at key points. Add these metric increments at the appropriate locations:

- After LLM call: `metrics.inc(LLM_CALLS_TOTAL, labels={"model": model_id})` + `metrics.observe(LLM_CALL_LATENCY, elapsed)`
- On cache hit: `metrics.inc(CACHE_HITS_TOTAL)`
- On cache miss: `metrics.inc(CACHE_MISSES_TOTAL)`
- On retry: `metrics.inc(VALIDATION_RETRIES_TOTAL)`
- On DLQ route: `metrics.inc(DLQ_ROUTES_TOTAL)`
- On review route: `metrics.inc(REVIEW_ROUTES_TOTAL)`
- On engineering route: `metrics.inc(ENGINEERING_ROUTES_TOTAL)`
- After eval: `metrics.observe(EVAL_SCORE, score)`

### Step 5: Add metrics API endpoint
Edit `src/api/main.py` to add:

```python
@app.get("/metrics")
async def get_metrics():
    from src.observability.metrics import Metrics
    return Metrics().snapshot()
```

### Step 6: Write tests
Create `tests/unit/test_observability.py`:

Test the Metrics class:
- `test_counter_increment` — inc by 1, by 5, check value
- `test_counter_with_labels` — inc with different labels, verify separation
- `test_histogram_observe` — observe values, check p50/p95/p99
- `test_timer_context_manager` — use timer(), verify observation recorded
- `test_snapshot_format` — verify snapshot returns expected structure
- `test_reset` — verify reset clears all data
- `test_singleton` — verify Metrics() returns same instance

Test the LangfuseTracker:
- `test_disabled_when_no_env` — no env vars → enabled=False, trace/generation still work (no-op)
- `test_trace_context_manager` — yields TraceSpan, records generations
- `test_generation_records_latency` — gen.end() populates latency_ms in metadata

### Step 7: Run all tests
```bash
pytest tests/ -v
```

### IMPORTANT NOTES
- The LangfuseTracker MUST work without langfuse installed (graceful ImportError handling)
- The Metrics class MUST work without prometheus_client (pure Python implementation)
- Read `src/pipeline/graph.py` carefully before adding metric calls — find the exact locations
- Don't add metrics everywhere — only at the 8 points listed in Step 4
- Keep total new code under 300 lines (tracker + metrics + tests)
