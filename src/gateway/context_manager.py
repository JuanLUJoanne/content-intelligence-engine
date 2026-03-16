"""
Context management helpers for prompt construction.

This module concentrates token accounting and truncation behaviour so that
callers can reason about safety margins in one place instead of sprinkling
ad‑hoc length checks across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import structlog

import tiktoken


logger = structlog.get_logger(__name__)


PromptVersion = Literal["short", "medium", "long"]


def _get_encoding(model: str):
    """Resolve a tiktoken encoding for the given model defensively."""

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        logger.info("tiktoken_fallback_encoding", model=model)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    """Return an approximate token count for `text` under `model`.

    Centralising this estimate makes it easier to keep safety margins in sync
    when provider tokenisation rules change.
    """

    if not text:
        return 0
    enc = _get_encoding(model)
    return len(enc.encode(text))


def truncate_to_fit(text: str, max_tokens: int, model: str) -> str:
    """Truncate text so that it fits within `max_tokens` for the model.

    The implementation preserves both the beginning and end of the string
    where possible, which tends to keep intent and key qualifiers intact
    compared to blunt prefix‑only truncation.
    """

    if max_tokens <= 0 or not text:
        return ""

    enc = _get_encoding(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Keep 70% of the budget for the prefix and 30% for the suffix.
    prefix_len = max_tokens * 7 // 10
    suffix_len = max_tokens - prefix_len
    prefix = tokens[:prefix_len]
    suffix = tokens[-suffix_len:] if suffix_len > 0 else []
    truncated = prefix + suffix
    decoded = enc.decode(truncated)
    return decoded


def select_prompt_version(complexity: float) -> PromptVersion:
    """Choose between short/medium/long prompt templates.

    Routing prompt length through this helper keeps changes to prompting
    strategy localised rather than hard‑coding thresholds at call sites.
    """

    c = max(0.0, min(complexity, 1.0))
    if c < 0.33:
        return "short"
    if c < 0.66:
        return "medium"
    return "long"


def compress_input(text: str, *, max_tokens: int, model: str) -> str:
    """Heuristically compress input while preserving salient details.

    This function avoids calling an LLM so that it can be used on hot paths
    or as a first‑pass guardrail before attempting more expensive semantic
    compression.
    """

    if not text:
        return text

    # Quick path: already cheap enough.
    if count_tokens(text, model) <= max_tokens:
        return text

    # Prefer keeping short lines (often titles, bullets) over dense blocks.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_lines = [ln for ln in lines if len(ln) <= 160]
    long_lines = [ln for ln in lines if len(ln) > 160]

    candidate_parts = short_lines + long_lines
    compressed: list[str] = []
    running_tokens = 0
    for part in candidate_parts:
        part_tokens = count_tokens(part, model)
        if running_tokens + part_tokens > max_tokens:
            break
        compressed.append(part)
        running_tokens += part_tokens

    if not compressed:
        # Fall back to raw truncation when even a single line is too large.
        return truncate_to_fit(text, max_tokens=max_tokens, model=model)

    result = "\n".join(compressed)
    if count_tokens(result, model) > max_tokens:
        result = truncate_to_fit(result, max_tokens=max_tokens, model=model)

    return result

