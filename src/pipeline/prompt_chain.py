"""
Prompt chaining pattern for multi-step LLM inference.

Chaining lets us decompose complex extraction tasks into focused steps where
each model call has a narrow, well-defined job. This improves accuracy
(smaller context → fewer hallucinations), makes debugging easier (failures are
localised to one step), and lets us cache intermediate results independently.

Step output is merged into the running context dict so later templates can
reference any field produced by earlier steps without manual wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Type

import structlog
from pydantic import BaseModel

from src.gateway.cache import CacheBackend, make_cache_key
from src.gateway.providers import LLMProvider


logger = structlog.get_logger(__name__)


@dataclass
class ChainStep:
    """One node in the prompt chain.

    prompt_template uses Python str.format() syntax; keys are drawn from the
    accumulated context dict (initial_input merged with all previous outputs).
    output_schema is validated with model_validate so schema violations surface
    at the step level rather than silently propagating bad data downstream.
    """

    name: str
    prompt_template: str
    output_schema: Type[BaseModel]
    max_retries: int = 3


@dataclass
class ChainStepResult:
    """Outcome of a single executed step."""

    step_name: str
    output: dict[str, Any]
    attempts: int
    cached: bool = False


@dataclass
class ChainResult:
    """Aggregate result of a full chain run."""

    steps: list[ChainStepResult]
    # Merged dict of initial_input plus all step outputs.
    final_output: dict[str, Any]
    total_latency: float


class PromptChain:
    """Execute a sequence of prompt steps, feeding each output into the next.

    Each step is independently retryable — a transient failure in step 2
    does not re-run step 1. Each step is also independently cacheable so
    re-running a chain after a partial failure skips completed steps.

    LangSmith integration: wrap each _run_step call with a langsmith.trace()
    context manager to surface individual steps as child spans in the trace UI.
    TODO: Add @traceable decorators once langsmith is added to dependencies.
    """

    def __init__(
        self,
        steps: list[ChainStep],
        provider: LLMProvider,
        *,
        model_id: str = "dummy",
        cache: CacheBackend | None = None,
    ) -> None:
        self._steps = steps
        self._provider = provider
        self._model_id = model_id
        self._cache = cache

    async def run(self, initial_input: dict[str, Any]) -> ChainResult:
        """Execute all steps in order, accumulating context between them."""
        context: dict[str, Any] = dict(initial_input)
        step_results: list[ChainStepResult] = []
        t0 = time.monotonic()

        for step in self._steps:
            result = await self._run_step(step, context)
            step_results.append(result)
            context.update(result.output)

        latency = time.monotonic() - t0
        logger.info(
            "chain_complete",
            steps=len(step_results),
            latency_s=round(latency, 3),
        )
        return ChainResult(steps=step_results, final_output=context, total_latency=latency)

    async def _run_step(
        self, step: ChainStep, context: dict[str, Any]
    ) -> ChainStepResult:
        """Try step up to max_retries times; only this step retries on error."""
        prompt = step.prompt_template.format(**context)
        cache_key = (
            make_cache_key(self._model_id, step.name, prompt)
            if self._cache is not None
            else None
        )

        if cache_key and self._cache:
            cached_str = await self._cache.get(cache_key)
            if cached_str is not None:
                logger.info("chain_step_complete", step=step.name, cached=True)
                return ChainStepResult(
                    step_name=step.name,
                    output=json.loads(cached_str),
                    attempts=0,
                    cached=True,
                )

        last_exc: BaseException | None = None
        for attempt in range(1, step.max_retries + 1):
            logger.info("chain_step_start", step=step.name, attempt=attempt)
            try:
                raw = await self._provider.generate(prompt, self._model_id)
                validated = step.output_schema.model_validate(raw)
                output = validated.model_dump(mode="json")

                if cache_key and self._cache:
                    await self._cache.set(cache_key, json.dumps(output), 3600)

                logger.info("chain_step_complete", step=step.name, attempt=attempt)
                return ChainStepResult(
                    step_name=step.name, output=output, attempts=attempt
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "chain_step_retry",
                    step=step.name,
                    attempt=attempt,
                    max_retries=step.max_retries,
                    error=str(exc),
                )

        raise RuntimeError(
            f"Step '{step.name}' failed after {step.max_retries} attempts"
        ) from last_exc
