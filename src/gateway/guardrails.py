"""
Spend and output guardrails for LLM usage.

This module layers additional protections on top of basic cost tracking so
that sudden traffic spikes, pricing changes, or prompt regressions cannot
silently drive runaway spend. Centralising these checks makes it easier to
codify organisational risk tolerance in one place.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Iterable, Optional, Tuple

import structlog

from .cost_tracker import CostTracker, Money


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GuardrailResult:
    """Return value of CostGuardrail.check() for non-hard-stop outcomes."""

    allowed: bool
    reason: str
    level: str  # "ok" | "per_minute" | "anomaly"


class CostGuardrail:
    """Single-method cost guard with multi-level protection.

    This class is intentionally simpler than CostGuardrailChain: it exposes
    one synchronous ``check()`` call that returns a GuardrailResult for soft
    blocks and raises GuardrailViolation for hard stops, making it easy to
    compose with any async or sync call site.

    Rolling average tracks the last 100 items for anomaly detection.
    """

    def __init__(
        self,
        *,
        per_request_limit: float = 0.05,
        per_minute_limit: float = 10.0,
        anomaly_multiplier: float = 3.0,
        total_budget_limit: float | None = None,
        cost_per_1k_tokens: float = 0.01,
    ) -> None:
        self._per_request_limit = per_request_limit
        self._per_minute_limit = per_minute_limit
        self._anomaly_multiplier = anomaly_multiplier
        self._total_budget_limit = total_budget_limit
        self._cost_per_1k = cost_per_1k_tokens

        self._minute_window: list[tuple[float, float]] = []
        self._rolling_costs: deque[float] = deque(maxlen=100)
        self._total_spent: float = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * self._cost_per_1k / 1000.0

    def _prune_minute_window(self, now: float) -> None:
        cutoff = now - 60.0
        self._minute_window = [(t, c) for t, c in self._minute_window if t >= cutoff]

    def _minute_total(self) -> float:
        return sum(c for _, c in self._minute_window)

    def _rolling_avg(self) -> float | None:
        if not self._rolling_costs:
            return None
        return sum(self._rolling_costs) / len(self._rolling_costs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, model_id: str, estimated_tokens: int) -> GuardrailResult:
        """Check whether the estimated call is within all configured limits.

        Records the call cost on success so that sliding-window and rolling
        average state stays accurate without requiring a separate method call.

        Returns:
            GuardrailResult with allowed=True when all limits pass.
            GuardrailResult with allowed=False for soft blocks (per_minute,
            anomaly); the caller should pause and retry.

        Raises:
            GuardrailViolation: for hard stops (per_request limit or total
            budget threshold reached), where retrying would not help.
        """
        estimated_cost = self._estimate_cost(estimated_tokens)
        now = time.monotonic()
        self._prune_minute_window(now)

        # --- Hard stop: per-request limit ---
        if estimated_cost > self._per_request_limit:
            logger.warning(
                "guardrail_triggered",
                level="per_request",
                model_id=model_id,
                estimated_cost=round(estimated_cost, 6),
                limit=self._per_request_limit,
                reason="per-request limit exceeded",
            )
            raise GuardrailViolation(
                f"Per-request cost ${estimated_cost:.4f} exceeds limit ${self._per_request_limit}"
            )

        # --- Hard stop: total budget at 95% ---
        if self._total_budget_limit is not None:
            threshold = self._total_budget_limit * 0.95
            if self._total_spent + estimated_cost >= threshold:
                logger.warning(
                    "guardrail_triggered",
                    level="budget",
                    model_id=model_id,
                    total_spent=round(self._total_spent, 6),
                    threshold=round(threshold, 6),
                    reason="total budget 95% threshold reached",
                )
                raise GuardrailViolation(
                    f"Total budget ${self._total_budget_limit} at 95% threshold "
                    f"(spent=${self._total_spent:.4f})"
                )

        # --- Soft block: per-minute limit ---
        if self._minute_total() + estimated_cost > self._per_minute_limit:
            result = GuardrailResult(
                allowed=False,
                reason=f"per-minute spend would exceed ${self._per_minute_limit}",
                level="per_minute",
            )
            logger.warning(
                "guardrail_triggered",
                level="per_minute",
                model_id=model_id,
                reason=result.reason,
            )
            return result

        # --- Soft block: anomaly detection ---
        avg = self._rolling_avg()
        if avg is not None and avg > 0 and estimated_cost > avg * self._anomaly_multiplier:
            result = GuardrailResult(
                allowed=False,
                reason=(
                    f"cost ${estimated_cost:.4f} is {self._anomaly_multiplier}x "
                    f"above rolling avg ${avg:.4f}"
                ),
                level="anomaly",
            )
            logger.warning(
                "guardrail_triggered",
                level="anomaly",
                model_id=model_id,
                reason=result.reason,
            )
            return result

        # --- Allowed: record and return ---
        self._minute_window.append((now, estimated_cost))
        self._rolling_costs.append(estimated_cost)
        self._total_spent += estimated_cost

        return GuardrailResult(allowed=True, reason="ok", level="ok")


class GuardrailViolation(RuntimeError):
    """Raised when a hard spend or safety limit is exceeded."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class _CostSample:
    timestamp: float
    cost: Money


class CostGuardrailChain:
    """Multi-level cost guard rails built around `CostTracker`.

    The chain's responsibility is to decide when to block or pause calls
    independent of any particular provider so that budget and anomaly
    thresholds can be tuned without rewriting business logic.
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        *,
        per_request_limit: Money | Decimal | float | str = Decimal("0.05"),
        per_minute_limit: Money | Decimal | float | str = Decimal("10.0"),
        per_hour_limit: Money | Decimal | float | str = Decimal("100.0"),
        total_budget_limit: Optional[Money | Decimal | float | str] = None,
    ) -> None:
        self._cost_tracker = cost_tracker
        self._per_request_limit = self._to_money(per_request_limit)
        self._per_minute_limit = self._to_money(per_minute_limit)
        self._per_hour_limit = self._to_money(per_hour_limit)
        self._total_budget_limit = (
            self._to_money(total_budget_limit) if total_budget_limit is not None else None
        )

        self._minute_samples: Deque[_CostSample] = deque()
        self._hour_samples: Deque[_CostSample] = deque()
        self._cost_history: Deque[Money] = deque(maxlen=500)
        self._lock = asyncio.Lock()

    @staticmethod
    def _to_money(value: Money | Decimal | float | str) -> Money:
        if isinstance(value, Decimal):
            return value
        return Money(str(value))

    def _prune_windows(self, now: float) -> None:
        minute_cutoff = now - 60.0
        hour_cutoff = now - 3600.0
        while self._minute_samples and self._minute_samples[0].timestamp < minute_cutoff:
            self._minute_samples.popleft()
        while self._hour_samples and self._hour_samples[0].timestamp < hour_cutoff:
            self._hour_samples.popleft()

    def _window_totals(self) -> Tuple[Money, Money]:
        minute_total = sum((s.cost for s in self._minute_samples), Money("0"))
        hour_total = sum((s.cost for s in self._hour_samples), Money("0"))
        return minute_total, hour_total

    def _moving_average_cost(self) -> Optional[Money]:
        if not self._cost_history:
            return None
        total = sum(self._cost_history, Money("0"))
        return total / Decimal(len(self._cost_history))

    async def check_and_record(
        self,
        *,
        model: str,
        cost: Money | Decimal | float | str,
        expected_output_tokens: int,
        actual_output_tokens: Optional[int] = None,
    ) -> None:
        """Apply guardrails for a single call and record its cost.

        This method should be called immediately after a successful request;
        it will raise `GuardrailViolation` when a hard stop is required or
        sleep briefly when soft limits are exceeded.
        """

        money_cost = self._to_money(cost)
        now = time.time()

        async with self._lock:
            # Per-request hard stop.
            if money_cost > self._per_request_limit:
                logger.error(
                    "guardrail_per_request_limit_exceeded",
                    model=model,
                    cost=str(money_cost),
                    limit=str(self._per_request_limit),
                )
                raise GuardrailViolation(
                    f"Per-request limit {self._per_request_limit} exceeded: {money_cost}"
                )

            self._minute_samples.append(_CostSample(timestamp=now, cost=money_cost))
            self._hour_samples.append(_CostSample(timestamp=now, cost=money_cost))
            self._cost_history.append(money_cost)
            self._prune_windows(now)

            minute_total, hour_total = self._window_totals()

            # Per-minute soft limit: brief pause but allow progress.
            if minute_total > self._per_minute_limit:
                logger.warning(
                    "guardrail_per_minute_limit_exceeded",
                    model=model,
                    minute_total=str(minute_total),
                    limit=str(self._per_minute_limit),
                )
                await asyncio.sleep(1.0)

            # Per-hour soft limit: longer pause and explicit notification.
            if hour_total > self._per_hour_limit:
                logger.error(
                    "guardrail_per_hour_limit_exceeded",
                    model=model,
                    hour_total=str(hour_total),
                    limit=str(self._per_hour_limit),
                )
                await asyncio.sleep(5.0)

            # Total-budget hard stop at 95% of configured cap, if we know it.
            if self._total_budget_limit is not None:
                remaining = self._cost_tracker.remaining_budget
                if remaining is not None:
                    spent = self._total_budget_limit - remaining
                    threshold = self._total_budget_limit * Decimal("0.95")
                    if spent >= threshold:
                        logger.error(
                            "guardrail_total_budget_threshold_reached",
                            spent=str(spent),
                            threshold=str(threshold),
                        )
                        raise GuardrailViolation(
                            f"Total budget threshold {threshold} reached (spent={spent})"
                        )

            # Anomaly detection on cost-per-item.
            avg = self._moving_average_cost()
            if avg is not None and avg > Money("0"):
                if money_cost > avg * Decimal("3"):
                    logger.warning(
                        "guardrail_anomalous_cost_per_item",
                        model=model,
                        cost=str(money_cost),
                        moving_average=str(avg),
                    )
                    await asyncio.sleep(2.0)

            # Output length guard: warn on suspiciously long responses.
            if actual_output_tokens is not None and expected_output_tokens > 0:
                if actual_output_tokens > expected_output_tokens * 2:
                    logger.warning(
                        "guardrail_output_length_exceeded",
                        model=model,
                        expected_tokens=expected_output_tokens,
                        actual_tokens=actual_output_tokens,
                    )

