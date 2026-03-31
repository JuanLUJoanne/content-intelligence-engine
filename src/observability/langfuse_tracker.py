"""
Langfuse LLM observability integration.

Wraps LLM calls with generation tracking: input, output, model, latency, cost, tokens.
Gracefully degrades to no-op when LANGFUSE_PUBLIC_KEY is not set.

Usage:
    tracker = LangfuseTracker()  # reads from env

    with tracker.trace("process_record", metadata={"record_id": "abc"}) as trace:
        with trace.generation("llm_call", model="gemini-2.0-flash", input_data=prompt) as gen:
            result = await provider.generate(prompt)
            gen.end(output=result, usage={"input_tokens": 100, "output_tokens": 50})

        with trace.generation("judge_eval", model="gpt-4o-mini", input_data=eval_prompt) as gen:
            score = await judge.evaluate(result)
            gen.end(output=score)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

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
    _langfuse_generation: Any = None

    def end(
        self,
        output: Any = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
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
    _langfuse_trace: Any = None
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
