"""
Runtime self-critique node for the content extraction pipeline.

Sits between schema validation and confidence scoring. The node asks the LLM
to critique its own extraction against three lightweight dimensions (factual
accuracy, hallucination, schema compliance) and re-extracts once if the
critique finds issues.

Gated behind the ``reflection_enabled`` feature flag so the extra LLM call
(and its latency + cost) can be A/B tested against the baseline pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from src.gateway.providers import LLMProvider
from src.observability.metrics import (
    LLM_CALLS_TOTAL,
    REFLECTION_CORRECTIONS_TOTAL,
    REFLECTION_TOTAL,
    Metrics,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Critique prompt (~400 tokens, well under the 500 limit)
# ---------------------------------------------------------------------------

_CRITIQUE_PROMPT = """\
You are a quality reviewer for product metadata extraction.

ORIGINAL INPUT:
{input_text}

EXTRACTED METADATA:
{extraction}

Check these three dimensions:
1. FACTUAL ACCURACY — Does the category match the content described? Is the condition assessment reasonable given the text?
2. HALLUCINATION — Are all tags actually mentioned or implied by the input? Is any field fabricated?
3. SCHEMA COMPLIANCE — Is price_range consistent with the description? Are enum values valid (category: electronics|clothing|home|sports|books, condition: new|like_new|good|fair|refurbished, price_range: budget|mid_range|premium|luxury|unpriced)?

Return JSON only:
{{"approved": true/false, "issues": ["issue1", ...], "suggestion": "corrected JSON or empty string"}}
"""

_REEXTRACT_PROMPT = """\
Re-extract metadata from this input. A reviewer found these issues with your previous extraction:

ISSUES:
{issues}

ORIGINAL INPUT:
{input_text}

PREVIOUS EXTRACTION:
{extraction}

Fix the identified issues and return ONLY a valid JSON object with keys: title, category, condition, price_range, tags, language, description.
"""


# ---------------------------------------------------------------------------
# Critique result
# ---------------------------------------------------------------------------


@dataclass
class CritiqueResult:
    """Structured output from the reflection node."""

    approved: bool
    issues: list[str]
    suggestion: str
    corrected_output: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Core reflection logic
# ---------------------------------------------------------------------------


def build_critique_prompt(input_text: str, extraction: Dict[str, Any]) -> str:
    """Build the critique prompt from input and extraction."""
    return _CRITIQUE_PROMPT.format(
        input_text=input_text,
        extraction=json.dumps(extraction, ensure_ascii=False),
    )


def build_reextract_prompt(
    input_text: str,
    extraction: Dict[str, Any],
    issues: list[str],
) -> str:
    """Build the re-extraction prompt incorporating reviewer feedback."""
    return _REEXTRACT_PROMPT.format(
        issues="\n".join(f"- {i}" for i in issues),
        input_text=input_text,
        extraction=json.dumps(extraction, ensure_ascii=False),
    )


def parse_critique(raw: Any) -> CritiqueResult:
    """Parse the LLM's critique response, fail-open on parse errors.

    If the response cannot be parsed, the extraction is treated as approved
    so the pipeline continues without blocking.
    """
    try:
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            data = json.loads(str(raw))

        approved = bool(data.get("approved", True))
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        suggestion = str(data.get("suggestion", ""))
        return CritiqueResult(
            approved=approved,
            issues=[str(i) for i in issues],
            suggestion=suggestion,
        )
    except Exception as exc:
        logger.warning("reflection_critique_parse_failed", error=str(exc))
        return CritiqueResult(approved=True, issues=[], suggestion="")


async def reflect(
    *,
    provider: LLMProvider,
    model_id: str,
    input_text: str,
    llm_output: Dict[str, Any],
    metrics: Optional[Metrics] = None,
) -> tuple[Dict[str, Any], CritiqueResult]:
    """Run self-critique and optionally re-extract.

    Returns
    -------
    (final_output, critique)
        The (possibly corrected) output and the critique result.
    """
    _metrics = metrics or Metrics()
    _metrics.inc(REFLECTION_TOTAL)

    # Step 1: Critique the extraction
    critique_prompt = build_critique_prompt(input_text, llm_output)
    try:
        raw_critique = await provider.generate(critique_prompt, model_id)
        _metrics.inc(LLM_CALLS_TOTAL, labels={"model": f"{model_id}_reflection"})
    except Exception as exc:
        logger.warning("reflection_critique_call_failed", error=str(exc))
        return llm_output, CritiqueResult(approved=True, issues=[], suggestion="")

    critique = parse_critique(raw_critique)

    if critique.approved:
        logger.info("reflection_approved", model=model_id)
        return llm_output, critique

    # Step 2: Re-extract with feedback (max 1 retry)
    logger.info(
        "reflection_issues_found",
        issues=critique.issues,
        model=model_id,
    )
    _metrics.inc(REFLECTION_CORRECTIONS_TOTAL)

    reextract_prompt = build_reextract_prompt(input_text, llm_output, critique.issues)
    try:
        raw_corrected = await provider.generate(reextract_prompt, model_id)
        _metrics.inc(LLM_CALLS_TOTAL, labels={"model": f"{model_id}_reflection"})
    except Exception as exc:
        logger.warning("reflection_reextract_failed", error=str(exc))
        return llm_output, critique

    # Parse the corrected output
    if isinstance(raw_corrected, dict):
        corrected = raw_corrected
    else:
        try:
            corrected = json.loads(str(raw_corrected))
        except Exception:
            logger.warning("reflection_corrected_parse_failed")
            return llm_output, critique

    critique.corrected_output = corrected
    logger.info("reflection_correction_applied", model=model_id)
    return corrected, critique
