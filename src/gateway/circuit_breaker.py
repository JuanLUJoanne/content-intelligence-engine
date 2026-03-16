"""
Circuit breaker for model backends.

This module introduces a small, explicit state machine so routing logic can
quickly decide when to stop sending traffic to a flaky model and when to
carefully let it back into rotation. Centralising this behaviour avoids
scattering ad‑hoc retry counters throughout the codebase and makes it easier
to reason about blast radius, recovery windows, and SLO trade‑offs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import structlog


logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """High‑level health of a backend integration.

    We keep the state compact and explicit so routing decisions are cheap and
    auditable. The goal is to prevent cascading failures when a model starts
    timing out or erroring in bursts.
    """

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreaker:
    """Track failure patterns for a single logical backend.

    The breaker exists to enforce a consistent policy for when to cut traffic,
    rather than letting each caller implement bespoke retry rules. This makes
    incident analysis and tuning far easier because we only have one dial to
    adjust per backend.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_success_threshold: int = 2
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: Optional[float] = field(default=None, init=False)
    half_open_success_count: int = field(default=0, init=False)

    def can_execute(self, now: Optional[float] = None) -> bool:
        """Return whether calls should be attempted for this backend.

        This check deliberately does not perform I/O so that routing can ask it
        on hot paths without adding latency; the caller remains responsible for
        actually recording success or failure.
        """

        if now is None:
            now = time.monotonic()

        if self.state is CircuitState.OPEN:
            if self.last_failure_time is not None and (
                now - self.last_failure_time >= self.recovery_timeout
            ):
                # Allow a trickle of traffic to probe whether the backend has
                # recovered, instead of flipping straight back to CLOSED.
                logger.info(
                    "circuit_transition_half_open",
                    name=self.name,
                    recovery_timeout=self.recovery_timeout,
                )
                self.state = CircuitState.HALF_OPEN
                self.half_open_success_count = 0
                return True

            # When the circuit is open and the timeout has not yet elapsed (or
            # we do not know when it last failed), we conservatively block
            # traffic so routing honours the explicit OPEN state.
            return False

        # CLOSED and HALF_OPEN both allow some traffic; HALF_OPEN is guarded by
        # success accounting below.
        return True

    def record_success(self) -> None:
        """Update breaker state after a successful call.

        The intent is to bias towards self‑healing behaviour: once a backend
        proves it can handle a small number of requests, we fully re‑enable it
        so the router can once again optimise for cost and latency instead of
        failure avoidance.
        """

        if self.state in (CircuitState.CLOSED,):
            self.failure_count = 0
            return

        if self.state is CircuitState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.half_open_success_threshold:
                logger.info(
                    "circuit_transition_closed",
                    name=self.name,
                    successes=self.half_open_success_count,
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None

    def record_failure(self, now: Optional[float] = None) -> None:
        """Update breaker state after a failed call.

        We treat any failure during HALF_OPEN as a strong signal that the
        backend is still unhealthy and immediately trip back to OPEN, forcing
        the router to rely on fallbacks.
        """

        if now is None:
            now = time.monotonic()

        self.failure_count += 1
        self.last_failure_time = now

        if self.state is CircuitState.HALF_OPEN:
            logger.warning(
                "circuit_retrip_from_half_open",
                name=self.name,
                failures=self.failure_count,
            )
            self.state = CircuitState.OPEN
            return

        if self.failure_count >= self.failure_threshold:
            if self.state is not CircuitState.OPEN:
                logger.warning(
                    "circuit_transition_open",
                    name=self.name,
                    failures=self.failure_count,
                    failure_threshold=self.failure_threshold,
                )
            self.state = CircuitState.OPEN

