"""
Input sanitization, PII detection, output validation, and audit logging.

This module layers security controls around LLM calls so that injection
attempts are blocked before they reach a model and sensitive data is
detected and redacted before it leaves the system. Centralising these
concerns here keeps individual pipeline nodes clean and makes the controls
easy to audit and update independently of business logic.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromptInjectionDetected(ValueError):
    """Raised when input text contains a recognised injection pattern."""

    def __init__(self, pattern: str) -> None:
        super().__init__(f"Prompt injection detected: matched pattern {pattern!r}")
        self.pattern = pattern


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------


_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore all instructions",
    "system prompt",
    "you are now",
    "pretend you are",
    "disregard all",
    "disregard previous",
    "jailbreak",
    "act as if you",
    "forget your instructions",
    "override your",
    "new instructions",
    "ignore previous",
)


class InputSanitizer:
    """Detect and block prompt injection attempts before they reach the LLM.

    Detection is case-insensitive substring matching against a curated list
    of phrases that appear in published jailbreak attempts. Centralising the
    list here makes it easy to add new patterns without touching pipeline code.
    """

    def sanitize(self, text: str) -> str:
        """Return the unchanged text if safe, or raise PromptInjectionDetected.

        The raw text is returned unmodified when no patterns match so callers
        do not need to handle a "cleaned" vs "raw" distinction.
        """
        lower = text.lower()
        for pattern in _INJECTION_PATTERNS:
            if pattern in lower:
                logger.warning(
                    "injection_detected",
                    pattern=pattern,
                    text_prefix=text[:80],
                )
                raise PromptInjectionDetected(pattern)
        return text


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------


@dataclass
class PIIFinding:
    """A single identified PII span within text."""

    type: str   # "email" | "phone" | "credit_card"
    value: str
    start: int
    end: int


_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
)
# Matches 16-digit card numbers with optional spaces or dashes between groups.
_CREDIT_CARD_RE = re.compile(
    r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL_RE),
    ("phone", _PHONE_RE),
    ("credit_card", _CREDIT_CARD_RE),
)


class PIIDetector:
    """Detect and redact personally identifiable information in text."""

    def detect(self, text: str) -> list[PIIFinding]:
        """Return all PII findings in position order."""
        findings: list[PIIFinding] = []
        for pii_type, pattern in _PII_PATTERNS:
            for m in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        type=pii_type,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        if findings:
            logger.warning(
                "pii_found",
                count=len(findings),
                types=sorted({f.type for f in findings}),
            )
        return sorted(findings, key=lambda f: f.start)

    def redact(self, text: str) -> str:
        """Replace each PII span with [REDACTED].

        Replacements are applied right-to-left so earlier span indices remain
        valid after each substitution.
        """
        findings = self.detect(text)
        for finding in reversed(findings):
            text = text[: finding.start] + "[REDACTED]" + text[finding.end :]
        return text


# ---------------------------------------------------------------------------
# OutputValidator
# ---------------------------------------------------------------------------


class OutputValidator:
    """Check LLM output for prompt leakage and PII before surfacing it.

    Two checks are applied:
    1. The output must not contain verbatim fragments of the system prompt
       (sliding-window length ≥ 10 chars), which would indicate leakage.
    2. The output must not contain PII patterns.
    """

    _CHUNK_SIZE = 10

    def __init__(self) -> None:
        self._pii = PIIDetector()

    def validate(self, output: str, system_prompt: str) -> bool:
        """Return True if the output is safe to surface, False otherwise."""
        # --- System-prompt leakage check ---
        step = max(self._CHUNK_SIZE // 2, 1)
        for i in range(0, len(system_prompt) - self._CHUNK_SIZE + 1, step):
            chunk = system_prompt[i : i + self._CHUNK_SIZE]
            if chunk.lower() in output.lower():
                logger.warning(
                    "output_validation_failed",
                    reason="system_prompt_fragment",
                )
                return False

        # --- PII check ---
        pii = self._pii.detect(output)
        if pii:
            logger.warning(
                "output_validation_failed",
                reason="pii_in_output",
                pii_count=len(pii),
            )
            return False

        return True


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Append-only JSONL audit trail for all processed items.

    Every processed request gets one line in the audit file regardless of
    success or failure so there is always a complete record for compliance
    and debugging without requiring a database.
    """

    def __init__(self, audit_path: str = "data/audit.jsonl") -> None:
        self._path = Path(audit_path)

    async def log(
        self,
        *,
        request_id: str,
        input: str,
        output: str,
        model: str,
        cost: float,
        timestamp: str,
    ) -> None:
        """Append one audit record; never truncates or overwrites the file."""
        entry = {
            "request_id": request_id,
            "input": input,
            "output": output,
            "model": model,
            "cost": cost,
            "timestamp": timestamp,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        await asyncio.to_thread(self._append_line, line)
        logger.info("audit_logged", request_id=request_id, model=model)

    def _append_line(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
