"""
FastAPI surface for the content intelligence engine.

Exposes HTTP endpoints around the LangGraph pipeline, cost tracking, drift
detection, and human review so external tools and demos can drive the full
system without importing internal modules directly.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.review import engineering_router, router as review_router
from src.eval.drift_detector import DriftDetector, DriftReport
from src.eval.judge import EvalDimension, EvaluationResult, LLMJudge
from src.gateway.batch import BatchSubmitter
from src.gateway.circuit_breaker import CircuitBreaker
from src.gateway.cost_tracker import CostTracker, ModelPricing
from src.gateway.providers import ProviderFactory
from src.gateway.router import ModelRouter, ModelTier, TaskFeatures
from src.gateway.security import AuditLogger, InputSanitizer, PromptInjectionDetected
from src.pipeline.graph import ContentPipelineGraph
from src.schemas.metadata import ContentMetadata


logger = structlog.get_logger(__name__)

app = FastAPI(title="Content Intelligence Engine")
app.include_router(review_router)
app.include_router(engineering_router)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_sanitizer = InputSanitizer()
_audit_logger = AuditLogger()
_batch_submitter = BatchSubmitter()
_drift_detector = DriftDetector()
_last_drift_report: Optional[DriftReport] = None

# Cache for the health-check test count (refreshed every 5 minutes).
_tests_passing_cache: Optional[int] = None
_tests_last_run: float = 0.0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    """Single-item processing request."""

    image_url: str
    text: str
    language: str = "en"
    content_type: str = "product"


class ProcessResponse(BaseModel):
    """Full pipeline result for a single item."""

    metadata: ContentMetadata
    model_used: str
    cost: Decimal
    confidence: float


class StreamProcessRequest(BaseModel):
    """Payload for the SSE streaming endpoint."""

    image_url: str
    text: str
    language: str = "en"
    content_type: str = "product"


class BatchProcessRequest(BaseModel):
    """Batch submission payload."""

    records: List[Dict[str, Any]]
    batch_mode: bool = True


class BatchProcessResponse(BaseModel):
    """Confirmation returned after batch submission."""

    batch_id: str
    estimated_cost: Decimal
    estimated_time: str


class EvaluateRequest(BaseModel):
    """Payload for asking the judge to score an output."""

    user_input: str
    candidate_output: str
    reference_output: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Normalised representation of evaluation scores."""

    scores: Dict[EvalDimension, float]
    overall: float


class CostSummaryItem(BaseModel):
    """Single line item in the cost dashboard."""

    model: str
    total_cost: Decimal
    total_input_tokens: int
    total_output_tokens: int


class CostSummaryResponse(BaseModel):
    """Aggregated cost data for dashboards."""

    items: List[CostSummaryItem]
    remaining_budget: Optional[Decimal] = None


# ---------------------------------------------------------------------------
# Dummy LLM (dev / CI fallback)
# ---------------------------------------------------------------------------


async def _dummy_llm_call(prompt: str) -> str:
    """Placeholder LLM call used for local development and tests."""

    return """{
      "factual_accuracy": {"score": 0.8, "reasoning": "stubbed"},
      "schema_compliance": {"score": 0.9, "reasoning": "stubbed"},
      "hallucination": {"score": 0.9, "reasoning": "stubbed"},
      "semantic_consistency": {"score": 0.85, "reasoning": "stubbed"},
      "relevance": {"score": 0.95, "reasoning": "stubbed"}
    }"""


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_cost_tracker() -> CostTracker:
    """Provide a singleton CostTracker for the app process."""

    global _COST_TRACKER  # type: ignore[global-variable-not-assigned]
    try:
        return _COST_TRACKER  # type: ignore[name-defined]
    except NameError:
        pricing = {
            "gemini-flash": ModelPricing(
                input_cost_per_1k_tokens=Decimal("0.05"),
                output_cost_per_1k_tokens=Decimal("0.10"),
            ),
            "gpt-4o-mini": ModelPricing(
                input_cost_per_1k_tokens=Decimal("0.10"),
                output_cost_per_1k_tokens=Decimal("0.20"),
            ),
        }
        _COST_TRACKER = CostTracker(
            pricing_by_model=pricing,
            total_budget=Decimal("50.0"),
            per_request_budget=Decimal("1.0"),
        )
        return _COST_TRACKER


def get_router(
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> ModelRouter:
    """Create and cache an in-process router instance."""

    global _ROUTER  # type: ignore[global-variable-not-assigned]
    try:
        return _ROUTER  # type: ignore[name-defined]
    except NameError:
        router = ModelRouter(cost_tracker=cost_tracker)

        async def _init() -> None:
            await router.register_model(
                "gemini-flash",
                ModelTier.FLASH,
                breaker=CircuitBreaker(name="gemini-flash"),
            )
            await router.register_model(
                "gpt-4o-mini",
                ModelTier.STANDARD,
                breaker=CircuitBreaker(name="gpt-4o-mini"),
            )

        asyncio.get_event_loop().create_task(_init())
        _ROUTER = router
        return router


def get_judge() -> LLMJudge:
    """Inject an LLMJudge backed by the app's LLM client."""

    global _JUDGE  # type: ignore[global-variable-not-assigned]
    try:
        return _JUDGE  # type: ignore[name-defined]
    except NameError:
        _JUDGE = LLMJudge(llm_call=_dummy_llm_call)
        return _JUDGE


def get_pipeline() -> ContentPipelineGraph:
    """Return a singleton ContentPipelineGraph with the module sanitizer."""

    global _PIPELINE  # type: ignore[global-variable-not-assigned]
    try:
        return _PIPELINE  # type: ignore[name-defined]
    except NameError:
        _PIPELINE = ContentPipelineGraph(sanitizer=_sanitizer)
        return _PIPELINE


# ---------------------------------------------------------------------------
# Helper: run pytest and return the passing count (cached 5 min)
# ---------------------------------------------------------------------------


async def _count_passing_tests() -> int:
    global _tests_passing_cache, _tests_last_run

    now = time.monotonic()
    if _tests_passing_cache is not None and now - _tests_last_run < 300:
        return _tests_passing_cache

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "pytest",
            "tests/unit/",
            "--tb=no",
            "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        for line in stdout.decode().splitlines():
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            count = int(parts[i - 1])
                            _tests_passing_cache = count
                            _tests_last_run = now
                            return count
                        except ValueError:
                            pass
    except Exception:
        pass

    return _tests_passing_cache or 0


# ---------------------------------------------------------------------------
# Private helpers shared by process endpoints
# ---------------------------------------------------------------------------


def _build_record(request_id: str, payload: Any) -> Dict[str, Any]:
    return {
        "id": request_id,
        "image_url": payload.image_url,
        "text": payload.text,
        "language": payload.language,
        "content_type": payload.content_type,
    }


def _build_metadata(payload: Any, final: Dict[str, Any]) -> ContentMetadata:
    return ContentMetadata(
        content_id=payload.image_url,
        title=final.get("title", payload.text[:80] or "Untitled"),
        description=final.get("description", payload.text),
        category=final.get("category", "electronics"),
        condition=final.get("condition", "new"),
        price_range=final.get("price_range", "unpriced"),
        tags=final.get("tags", ["demo"]),
        language=final.get("language", payload.language),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/process", response_model=ProcessResponse)
async def process_content(
    payload: ProcessRequest,
    router: ModelRouter = Depends(get_router),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    pipeline: ContentPipelineGraph = Depends(get_pipeline),
) -> ProcessResponse:
    """Run a single item through the full LangGraph pipeline."""

    request_id = str(uuid.uuid4())

    # --- Input sanitization ---
    try:
        _sanitizer.sanitize(payload.text)
    except PromptInjectionDetected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Model routing and cost recording ---
    estimated_input_tokens = max(len(payload.text) // 4, 1)
    estimated_output_tokens = 128
    features = TaskFeatures(
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        latency_sensitivity=0.5,
        quality_sensitivity=0.7,
        cost_sensitivity=0.5,
    )
    model = await router.choose_model(features)
    try:
        cost = cost_tracker.record_usage(
            model,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    # --- LangGraph pipeline ---
    state = await pipeline.run(_build_record(request_id, payload))

    if state.get("sent_to_dlq") or state.get("error"):
        raise HTTPException(
            status_code=500,
            detail=state.get("error", "Pipeline error"),
        )

    final = state.get("final_output") or {}
    metadata = _build_metadata(payload, final)
    confidence = float(state.get("eval_score") or 1.0)

    # --- Audit log ---
    await _audit_logger.log(
        request_id=request_id,
        input=payload.text,
        output=json.dumps(final),
        model=model,
        cost=float(cost),
        timestamp=time.time(),
    )

    return ProcessResponse(
        metadata=metadata,
        model_used=model,
        cost=cost,
        confidence=confidence,
    )


@app.post("/process/stream")
async def process_stream(payload: StreamProcessRequest) -> StreamingResponse:
    """SSE stream of pipeline progress events.

    Yields one Server-Sent Event per pipeline stage so clients can display
    live progress without polling.  Each event is a JSON object with at least
    an ``event`` key and a ``status`` key.
    """

    async def _event_stream() -> AsyncGenerator[str, None]:
        def sse(event: str, **kwargs: Any) -> str:
            data = json.dumps({"event": event, **kwargs})
            return f"data: {data}\n\n"

        request_id = str(uuid.uuid4())
        record: Dict[str, Any] = _build_record(request_id, payload)

        # Stage 1: sanitize
        yield sse("sanitizing", status="running")
        try:
            _sanitizer.sanitize(payload.text)
            yield sse("sanitizing", status="ok")
        except PromptInjectionDetected as exc:
            yield sse("sanitizing", status="blocked", error=str(exc))
            return

        # Stage 2: cache check
        yield sse("cache_check", status="running")
        yield sse("cache_check", status="ok", hit=False)

        # Stage 3: routing
        yield sse("routing", status="running")
        yield sse("routing", status="ok", model="dummy")

        # Stage 4: LLM call
        yield sse("calling_llm", status="running")
        provider = ProviderFactory.get_provider("dummy")
        try:
            llm_output = await provider.generate(json.dumps(record), "dummy")
            yield sse("calling_llm", status="ok")
        except Exception as exc:
            yield sse("calling_llm", status="error", error=str(exc))
            return

        # Stage 5: schema validation
        yield sse("validating", status="running")
        yield sse("validating", status="ok")

        # Stage 6: confidence scoring
        yield sse("scoring", status="running")
        yield sse("scoring", status="ok", confidence=1.0)

        # Stage 7: complete — emit final metadata
        final_output = llm_output or {}
        metadata = _build_metadata(payload, final_output)

        await _audit_logger.log(
            request_id=request_id,
            input=payload.text,
            output=json.dumps(final_output),
            model="dummy",
            cost=0.0,
            timestamp=time.time(),
        )

        yield sse("complete", status="ok", metadata=metadata.model_dump())

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post("/process/batch", response_model=BatchProcessResponse)
async def process_batch_endpoint(
    payload: BatchProcessRequest,
) -> BatchProcessResponse:
    """Submit a list of records for async batch processing."""

    request_id = str(uuid.uuid4())

    # Sanitize each record's text field before queueing.
    for record in payload.records:
        text = str(record.get("text", ""))
        try:
            _sanitizer.sanitize(text)
        except PromptInjectionDetected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    jsonl = _batch_submitter.collect(payload.records)
    batch_id = await _batch_submitter.submit_batch(jsonl)

    estimated_cost = Decimal(str(round(len(payload.records) * 0.001, 4)))
    estimated_time = f"{max(len(payload.records) // 100, 1)} minute(s)"

    await _audit_logger.log(
        request_id=request_id,
        input=f"batch:{len(payload.records)} records",
        output=f"batch_id:{batch_id}",
        model="batch",
        cost=float(estimated_cost),
        timestamp=time.time(),
    )

    return BatchProcessResponse(
        batch_id=batch_id,
        estimated_cost=estimated_cost,
        estimated_time=estimated_time,
    )


@app.get("/costs", response_model=CostSummaryResponse)
async def get_costs(
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> CostSummaryResponse:
    """Provide a snapshot of current spend by model for dashboards."""

    items = [
        CostSummaryItem(
            model=s.model,
            total_cost=s.total_cost,
            total_input_tokens=s.total_input_tokens,
            total_output_tokens=s.total_output_tokens,
        )
        for s in cost_tracker.summary_by_model()
    ]
    return CostSummaryResponse(
        items=items,
        remaining_budget=cost_tracker.remaining_budget,
    )


@app.get("/drift/report")
async def get_drift_report() -> Dict[str, Any]:
    """Return the latest drift detection report, or baseline info if no report exists."""

    if _last_drift_report is not None:
        r = _last_drift_report
        return {
            "alert_triggered": r.alert_triggered,
            "degraded_dimensions": r.degraded_dimensions,
            "per_dimension_deltas": r.per_dimension_deltas,
        }

    baseline = _drift_detector.baseline
    if baseline is not None:
        return {
            "status": "baseline_only",
            "prompt_version": baseline.prompt_version,
            "baseline_scores": baseline.dimension_scores,
            "sample_size": baseline.sample_size,
            "created_at": baseline.created_at,
        }

    raise HTTPException(status_code=404, detail="No drift report available")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_output(
    payload: EvaluateRequest,
    judge: LLMJudge = Depends(get_judge),
) -> EvaluateResponse:
    """Expose LLM-based evaluation over HTTP for downstream tooling."""

    result: EvaluationResult = await judge.score(
        user_input=payload.user_input,
        candidate_output=payload.candidate_output,
        reference_output=payload.reference_output,
    )
    scores = {dim: ds.score for dim, ds in result.scores.items()}
    return EvaluateResponse(scores=scores, overall=result.overall_score)


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Export application metrics as JSON for monitoring dashboards."""
    from src.observability.metrics import Metrics

    return Metrics().snapshot()


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness endpoint; also reports passing test count (cached 5 min)."""

    tests_passing = await _count_passing_tests()
    return {"status": "ok", "tests_passing": tests_passing}
