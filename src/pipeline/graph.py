"""
LangGraph-style orchestration pipeline for content metadata extraction.

State flows through seven nodes with conditional edges for cache hits,
validation retries, DLQ routing, and human review. Each node is a method
on ``ContentPipelineGraph`` so they are easy to unit-test via mock injection.

LangSmith tracing is applied to ``run()`` via ``@traceable`` when the
``langsmith`` package is present. Without an API key the decorator is a
transparent no-op.

To migrate to actual LangGraph:
    1. ``pip install langgraph``
    2. Build a ``StateGraph(ProcessingState)``
    3. Add each ``_node_*`` method as a node
    4. Wire conditional edges to match the predicates in ``run()``
    5. Call ``graph.compile().ainvoke(initial_state)``
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional, Protocol, TypedDict, runtime_checkable

import structlog

from src.gateway.providers import LLMProvider, ProviderFactory
from src.schemas.metadata import ContentMetadata


try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(func: Any = None, **_kw: Any) -> Any:
        if func is not None:
            return func
        def _w(f: Any) -> Any:
            return f
        return _w


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ProcessingState TypedDict
# ---------------------------------------------------------------------------


class ProcessingState(TypedDict):
    """Full state dict passed between graph nodes."""

    record: dict[str, Any]
    cache_result: Optional[str]          # JSON string on cache hit
    model_id: Optional[str]
    llm_output: Optional[dict[str, Any]]
    validation_result: Optional[bool]
    eval_score: Optional[float]
    retry_count: int
    final_output: Optional[dict[str, Any]]
    sent_to_dlq: bool
    sent_to_review: bool
    error: Optional[str]


def _initial_state(record: dict[str, Any]) -> ProcessingState:
    return ProcessingState(
        record=record,
        cache_result=None,
        model_id=None,
        llm_output=None,
        validation_result=None,
        eval_score=None,
        retry_count=0,
        final_output=None,
        sent_to_dlq=False,
        sent_to_review=False,
        error=None,
    )


def _cache_key(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CacheBackend protocol (avoids importing the concrete module)
# ---------------------------------------------------------------------------


@runtime_checkable
class _CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str, ttl: int) -> None: ...


# ---------------------------------------------------------------------------
# Default schema validator
# ---------------------------------------------------------------------------


def _default_validate(record: dict[str, Any], llm_output: dict[str, Any]) -> bool:
    """Attempt ContentMetadata construction; return True on success."""
    try:
        ContentMetadata(
            content_id=str(record.get("id", "")),
            title=llm_output.get("title", ""),
            description=llm_output.get("description"),
            category=llm_output.get("category", "electronics"),
            condition=llm_output.get("condition", "new"),
            price_range=llm_output.get("price_range", "unpriced"),
            tags=llm_output.get("tags", []),
            language=llm_output.get("language", "en"),
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ContentPipelineGraph
# ---------------------------------------------------------------------------


class ContentPipelineGraph:
    """Seven-node content pipeline with conditional routing.

    Execution order::

        sanitize_input
            ↓
        check_cache ──(hit)──→ store_result
            ↓
        route_model
            ↓
        call_llm ←──────────────────────────(retry)──┐
            ↓                                         │
        validate_schema ──(fail, retries left)────────┘
            ↓ (fail, max retries)
        send_to_dlq
            ↓ (pass)
        score_confidence ──(score < threshold)──→ send_to_review
            ↓
        store_result
    """

    _MAX_RETRIES = 3

    def __init__(
        self,
        *,
        cache: Optional[_CacheBackend] = None,
        provider: Optional[LLMProvider] = None,
        model_id: str = "dummy",
        sanitizer: Any = None,
        validate_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
        confidence_fn: Optional[Callable[[dict[str, Any]], float]] = None,
        confidence_threshold: float = 0.7,
        dlq: Optional[list[ProcessingState]] = None,
        review_queue: Optional[list[ProcessingState]] = None,
    ) -> None:
        self._cache = cache
        self._provider = provider or ProviderFactory.get_provider(model_id)
        self._model_id = model_id
        self._sanitizer = sanitizer
        self._validate_fn = validate_fn
        self._confidence_fn = confidence_fn or (lambda _: 1.0)
        self._confidence_threshold = confidence_threshold
        self._dlq: list[ProcessingState] = dlq if dlq is not None else []
        self._review_queue: list[ProcessingState] = (
            review_queue if review_queue is not None else []
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @_traceable
    async def run(self, record: dict[str, Any]) -> ProcessingState:
        """Execute the pipeline for one record and return the final state."""
        state = _initial_state(record)

        # Node 1: sanitize input
        state = await self._node_sanitize_input(state)
        if state["error"]:
            return state

        # Node 2: check cache
        state = await self._node_check_cache(state)
        if state["cache_result"] is not None:
            return await self._node_store_result(state)

        # Node 3: route model
        state = await self._node_route_model(state)

        # Nodes 4-5: call LLM + validate, with retry loop
        while True:
            state = await self._node_call_llm(state)
            if state["sent_to_dlq"] or state["error"]:
                return state

            state = await self._node_validate_schema(state)

            if state["validation_result"] is True:
                break

            if state["retry_count"] >= self._MAX_RETRIES:
                logger.warning(
                    "graph_max_retries_exceeded",
                    retry_count=state["retry_count"],
                    record_id=str(record.get("id", "")),
                )
                return await self._node_send_to_dlq(state, reason="max_retries_exceeded")

            state = {**state, "retry_count": state["retry_count"] + 1}
            logger.info("graph_retry", retry_count=state["retry_count"])

        # Node 6: score confidence
        state = await self._node_score_confidence(state)
        if (
            state["eval_score"] is not None
            and state["eval_score"] < self._confidence_threshold
        ):
            return await self._node_send_to_review(state)

        # Node 7: store result
        return await self._node_store_result(state)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def _node_sanitize_input(self, state: ProcessingState) -> ProcessingState:
        if self._sanitizer is None:
            return state
        try:
            self._sanitizer.sanitize(json.dumps(state["record"]))
        except Exception as exc:
            logger.warning("graph_injection_blocked", error=str(exc))
            return {**state, "error": str(exc), "sent_to_dlq": True}
        return state

    async def _node_check_cache(self, state: ProcessingState) -> ProcessingState:
        if self._cache is None:
            return state
        key = _cache_key(state["record"])
        cached = await self._cache.get(key)
        if cached is not None:
            logger.info("graph_cache_hit", record_id=str(state["record"].get("id", "")))
            return {**state, "cache_result": cached}
        return state

    async def _node_route_model(self, state: ProcessingState) -> ProcessingState:
        logger.info("graph_route_model", model_id=self._model_id)
        return {**state, "model_id": self._model_id}

    async def _node_call_llm(self, state: ProcessingState) -> ProcessingState:
        prompt = json.dumps(state["record"], ensure_ascii=False)
        try:
            result = await self._provider.generate(
                prompt, state["model_id"] or self._model_id
            )
            logger.info("graph_llm_called", model_id=state["model_id"])
            return {**state, "llm_output": result}
        except Exception as exc:
            logger.error("graph_llm_error", error=str(exc))
            return {**state, "error": str(exc), "sent_to_dlq": True}

    async def _node_validate_schema(self, state: ProcessingState) -> ProcessingState:
        llm_output = state["llm_output"] or {}
        try:
            if self._validate_fn is not None:
                valid = self._validate_fn(llm_output)
            else:
                valid = _default_validate(state["record"], llm_output)
        except Exception:
            valid = False

        logger.info(
            "graph_validate_schema",
            valid=valid,
            retry_count=state["retry_count"],
        )
        return {**state, "validation_result": valid}

    async def _node_score_confidence(self, state: ProcessingState) -> ProcessingState:
        score = self._confidence_fn(state["llm_output"] or {})
        logger.info("graph_confidence_scored", score=score)
        return {**state, "eval_score": score}

    async def _node_store_result(self, state: ProcessingState) -> ProcessingState:
        if state["cache_result"] is not None:
            try:
                final: dict[str, Any] = json.loads(state["cache_result"])
            except Exception:
                final = {"raw": state["cache_result"]}
        else:
            final = state["llm_output"] or {}

        if self._cache is not None and final and state["cache_result"] is None:
            key = _cache_key(state["record"])
            try:
                await self._cache.set(key, json.dumps(final, ensure_ascii=False), 3600)
            except Exception:
                pass

        logger.info(
            "graph_result_stored",
            from_cache=state["cache_result"] is not None,
        )
        return {**state, "final_output": final}

    async def _node_send_to_dlq(
        self, state: ProcessingState, *, reason: str = "unknown"
    ) -> ProcessingState:
        updated: ProcessingState = {**state, "sent_to_dlq": True, "error": reason}
        self._dlq.append(updated)
        logger.warning("graph_sent_to_dlq", reason=reason)
        return updated

    async def _node_send_to_review(self, state: ProcessingState) -> ProcessingState:
        updated: ProcessingState = {**state, "sent_to_review": True}
        self._review_queue.append(updated)
        logger.info("graph_sent_to_review", eval_score=state["eval_score"])
        return updated
