"""
LLM‑based evaluation primitives for pipeline quality monitoring.

This module isolates how we ask models to grade their own outputs so that
evaluation criteria evolve independently from business logic. Centralising the
dimensions and scoring schemes makes it much easier to compare runs over time
and to roll out new prompts without rewriting downstream analytics.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional

import structlog
from pydantic import BaseModel, Field, ValidationError


logger = structlog.get_logger(__name__)


class EvalDimension(str, Enum):
    """Axes along which we judge model outputs.

    These dimensions were chosen to mirror the practical failure modes seen in
    production LLM pipelines: drifting away from facts, breaking schemas,
    inventing content, losing semantic intent, or going off‑topic entirely.
    """

    FACTUAL_ACCURACY = "factual_accuracy"
    SCHEMA_COMPLIANCE = "schema_compliance"
    HALLUCINATION = "hallucination"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    RELEVANCE = "relevance"


class DimensionScore(BaseModel):
    """Fine‑grained score for a single evaluation dimension.

    Keeping reasoning alongside the numeric score makes it much easier to
    diagnose regressions when a new model or prompt underperforms.
    """

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class EvaluationResult(BaseModel):
    """Aggregate view of a judgment across all dimensions."""

    scores: Dict[EvalDimension, DimensionScore]

    @property
    def overall_score(self) -> float:
        """Return an unweighted mean across all dimensions.

        Using a simple mean keeps the headline metric stable and easy to
        explain; if callers need more nuanced weighting they can compute it on
        top of this structure.
        """

        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)


LLMCallable = Callable[[str], Awaitable[str]]


# Per-dimension prompt templates. Each template must produce a JSON object with
# exactly two keys: "score" (float 0-1) and "reasoning" (str). Keeping them
# here means wording can be tuned without touching any business logic.
_DIMENSION_PROMPTS: dict[str, str] = {
    EvalDimension.FACTUAL_ACCURACY.value: (
        "You are an evaluation model. Score the following output for FACTUAL ACCURACY.\n\n"
        "USER INPUT:\n{user_input}\n\n"
        "MODEL OUTPUT:\n{candidate_output}\n\n"
        "REFERENCE:\n{reference_output}\n\n"
        "Factual accuracy: Are factual claims correct and well supported? "
        "1.0 = completely accurate, 0.0 = completely wrong.\n\n"
        'Return JSON only: {{"score": <float 0-1>, "reasoning": "<brief reason>"}}'
    ),
    EvalDimension.HALLUCINATION.value: (
        "You are an evaluation model. Score the following output for HALLUCINATION (absence thereof).\n\n"
        "USER INPUT:\n{user_input}\n\n"
        "MODEL OUTPUT:\n{candidate_output}\n\n"
        "REFERENCE:\n{reference_output}\n\n"
        "Hallucination: Does the answer avoid inventing unsupported details? "
        "1.0 = no hallucination, 0.0 = heavy hallucination.\n\n"
        'Return JSON only: {{"score": <float 0-1>, "reasoning": "<brief reason>"}}'
    ),
    EvalDimension.SEMANTIC_CONSISTENCY.value: (
        "You are an evaluation model. Score the following output for SEMANTIC CONSISTENCY.\n\n"
        "USER INPUT:\n{user_input}\n\n"
        "MODEL OUTPUT:\n{candidate_output}\n\n"
        "REFERENCE:\n{reference_output}\n\n"
        "Semantic consistency: Does the output preserve the intent of the input? "
        "1.0 = perfectly consistent, 0.0 = entirely inconsistent.\n\n"
        'Return JSON only: {{"score": <float 0-1>, "reasoning": "<brief reason>"}}'
    ),
    EvalDimension.RELEVANCE.value: (
        "You are an evaluation model. Score the following output for RELEVANCE.\n\n"
        "USER INPUT:\n{user_input}\n\n"
        "MODEL OUTPUT:\n{candidate_output}\n\n"
        "REFERENCE:\n{reference_output}\n\n"
        "Relevance: Does the answer stay on topic and avoid unnecessary digressions? "
        "1.0 = completely relevant, 0.0 = completely off-topic.\n\n"
        'Return JSON only: {{"score": <float 0-1>, "reasoning": "<brief reason>"}}'
    ),
    EvalDimension.SCHEMA_COMPLIANCE.value: (
        "You are an evaluation model. Score the following output for SCHEMA COMPLIANCE.\n\n"
        "USER INPUT:\n{user_input}\n\n"
        "MODEL OUTPUT:\n{candidate_output}\n\n"
        "REFERENCE:\n{reference_output}\n\n"
        "Schema compliance: Does the output follow the expected format and constraints? "
        "1.0 = fully compliant, 0.0 = completely non-compliant.\n\n"
        'Return JSON only: {{"score": <float 0-1>, "reasoning": "<brief reason>"}}'
    ),
}

_FALLBACK_SCORE = 0.5
_FALLBACK_REASONING = "LLM returned invalid JSON after retry; using default score"


class LLMJudge:
    """Ask an LLM to grade another model's output along multiple dimensions.

    Each dimension is evaluated in a separate, focused LLM call so that the
    judge can apply targeted criteria. Invalid JSON responses are retried once;
    if the retry also fails the dimension receives a neutral default score of
    0.5 so that one bad call cannot block the whole evaluation.

    Use ``LLMJudge.from_provider()`` for zero-config instantiation that
    requires no API keys (defaults to DummyProvider).
    """

    def __init__(self, llm_call: LLMCallable, *, model_id: str = "unknown") -> None:
        self._llm_call = llm_call
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_provider(cls, model_id: str = "dummy") -> "LLMJudge":
        """Create an LLMJudge backed by ProviderFactory (DummyProvider by default).

        This is the recommended entry point for tests and local development
        because it works without any API keys.
        """
        from src.gateway.providers import ProviderFactory

        provider = ProviderFactory.get_provider(model_id)

        async def _llm_call(prompt: str) -> str:
            result = await provider.generate(prompt, model_id)
            return json.dumps(result)

        return cls(_llm_call, model_id=model_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        *,
        user_input: str,
        candidate_output: str,
        reference_output: Optional[str] = None,
    ) -> str:
        """Construct a multi-dimension scoring prompt (legacy, kept for compatibility)."""

        dimensions = [d.value for d in EvalDimension]

        reference_block = (
            f"\nREFERENCE OUTPUT (optional, may be empty):\n{reference_output}\n"
            if reference_output is not None
            else "\nREFERENCE OUTPUT: <none provided>\n"
        )

        return f"""
You are an evaluation model scoring another model's response.

USER INPUT:
{user_input}

MODEL OUTPUT TO EVALUATE:
{candidate_output}
{reference_block}

For each of the following dimensions:
- factual_accuracy: Are factual claims correct and well supported?
- schema_compliance: Does the output follow the expected format and constraints?
- hallucination: Does the answer avoid inventing unsupported details?
- semantic_consistency: Does the output preserve the intent of the input?
- relevance: Does the answer stay on topic and avoid unnecessary digressions?

Return a single JSON object with this shape:
{{
  "factual_accuracy": {{"score": float between 0 and 1, "reasoning": str}},
  "schema_compliance": {{"score": float between 0 and 1, "reasoning": str}},
  "hallucination": {{"score": float between 0 and 1, "reasoning": str}},
  "semantic_consistency": {{"score": float between 0 and 1, "reasoning": str}},
  "relevance": {{"score": float between 0 and 1, "reasoning": str}}
}}

Do not include any text outside of the JSON.
"""

    async def _judge_dimension(
        self,
        dimension: EvalDimension,
        *,
        user_input: str,
        candidate_output: str,
        reference_output: Optional[str] = None,
        retry_delay: float = 2.0,
    ) -> DimensionScore:
        """Score a single dimension with one retry and a safe fallback.

        On the first attempt we call the LLM and try to parse the response.
        If parsing fails we wait ``retry_delay`` seconds then retry once. If the
        second attempt also fails we return a neutral score of 0.5 so that
        downstream aggregations degrade gracefully rather than raising an exception.
        """
        prompt = _DIMENSION_PROMPTS[dimension.value].format(
            user_input=user_input,
            candidate_output=candidate_output,
            reference_output=reference_output or "<none provided>",
        )

        last_error: str = ""
        for attempt in range(2):
            try:
                raw = await self._llm_call(prompt)
                if not isinstance(raw, str):
                    raw = json.dumps(raw)
                data = json.loads(raw)
                score_val = float(data["score"])
                if not (0.0 <= score_val <= 1.0):
                    raise ValueError(f"score {score_val} out of [0, 1]")
                result = DimensionScore(
                    score=score_val,
                    reasoning=str(data.get("reasoning", "")),
                )
                logger.info(
                    "eval_dimension_scored",
                    dimension=dimension.value,
                    score=score_val,
                    model=self._model_id,
                )
                return result

            except Exception as exc:
                last_error = str(exc)
                if attempt == 0:
                    logger.warning(
                        "eval_dimension_retry",
                        dimension=dimension.value,
                        error=last_error,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(
                        "eval_dimension_fallback",
                        dimension=dimension.value,
                        error=last_error,
                    )

        # Default after exhausted retries.
        logger.info(
            "eval_dimension_scored",
            dimension=dimension.value,
            score=_FALLBACK_SCORE,
            model=self._model_id,
            fallback=True,
        )
        return DimensionScore(score=_FALLBACK_SCORE, reasoning=_FALLBACK_REASONING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        *,
        user_input: str,
        candidate_output: str,
        reference_output: Optional[str] = None,
        inter_call_delay: float = 0.0,
    ) -> EvaluationResult:
        """Evaluate candidate_output along every dimension and return an aggregate.

        Each dimension is judged in a separate, sequential LLM call so that
        targeted criteria can be applied precisely without bursting the API
        rate limit. ``inter_call_delay`` (seconds) is inserted between calls;
        set this to match your ``--delay`` value when using a rate-limited
        provider. Invalid responses default to 0.5 rather than propagating
        exceptions.
        """
        dimensions = list(EvalDimension)
        scored: list[DimensionScore] = []
        for i, dim in enumerate(dimensions):
            result = await self._judge_dimension(
                dim,
                user_input=user_input,
                candidate_output=candidate_output,
                reference_output=reference_output,
                retry_delay=max(inter_call_delay, 2.0),
            )
            scored.append(result)
            if inter_call_delay > 0 and i < len(dimensions) - 1:
                await asyncio.sleep(inter_call_delay)
        scores: Dict[EvalDimension, DimensionScore] = dict(zip(dimensions, scored))
        return EvaluationResult(scores=scores)
