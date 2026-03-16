"""
A/B prompt comparison utilities.

This module provides a small harness for running controlled experiments between
two prompts and summarising the results statistically so that prompt changes
are guided by evidence instead of anecdotes.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import structlog

from .judge import EvalDimension, EvaluationResult, LLMJudge


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PromptTestCase:
    """Single input/expected pair used for A/B evaluation.

    Keeping test cases structured (rather than raw strings) allows future
    extensions like per‑case weights or tags without changing the comparator
    surface.
    """

    user_input: str
    reference_output: Optional[str] = None


@dataclass
class ABStatistics:
    """Summary statistics comparing two prompts along one dimension."""

    mean_a: float
    mean_b: float
    delta: float
    effect_size: float


@dataclass
class ABComparisonResult:
    """Full comparison outcome across all evaluation dimensions."""

    per_dimension: Dict[EvalDimension, ABStatistics]

    def best_prompt(self) -> Optional[str]:
        """Return 'A' or 'B' if one clearly dominates, or None if too close to call.

        A result is considered meaningful only when the overall mean delta
        exceeds 0.02; smaller differences are treated as noise given typical
        LLM score variance.
        """

        if not self.per_dimension:
            return None

        mean_a = statistics.mean(s.mean_a for s in self.per_dimension.values())
        mean_b = statistics.mean(s.mean_b for s in self.per_dimension.values())
        if abs(mean_a - mean_b) < 0.02:
            return None
        return "A" if mean_a > mean_b else "B"


class PromptComparator:
    """Run A/B prompt tests using an injected `LLMJudge`.

    Separating this from the main pipeline lets teams iterate quickly on
    wording and system instructions for specific tasks, while reusing the same
    evaluation criteria and logging infrastructure.
    """

    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    async def _score_prompt_for_case(
        self,
        prompt_template: str,
        case: PromptTestCase,
    ) -> EvaluationResult:
        """Apply a prompt template to a case and score the result."""

        user_input = prompt_template.format(input=case.user_input)
        # In many real setups you would call a *different* model here and then
        # feed its output into the judge. We keep it generic so that the caller
        # controls where generations come from.
        generated = case.reference_output or ""
        return await self._judge.score(
            user_input=user_input,
            candidate_output=generated,
            reference_output=case.reference_output,
        )

    @staticmethod
    def _compute_effect_size(a: Sequence[float], b: Sequence[float]) -> float:
        """Return Cohen's d between two score distributions."""

        if not a or not b:
            return 0.0

        mean_a = statistics.mean(a)
        mean_b = statistics.mean(b)
        var_a = statistics.pvariance(a)
        var_b = statistics.pvariance(b)
        pooled_var = (var_a + var_b) / 2
        if pooled_var == 0:
            return 0.0
        return (mean_b - mean_a) / (pooled_var**0.5)

    async def compare(
        self,
        *,
        prompt_a: str,
        prompt_b: str,
        cases: Iterable[PromptTestCase],
    ) -> ABComparisonResult:
        """Run an A/B test over a set of cases and summarise the result.

        The comparator keeps orchestration logic here so that tests stay
        reproducible; callers only need to provide the prompts and the corpus
        of examples that best represent their workload.
        """

        cases_list = list(cases)
        pairs = await asyncio.gather(*[
            asyncio.gather(
                self._score_prompt_for_case(prompt_template=prompt_a, case=case),
                self._score_prompt_for_case(prompt_template=prompt_b, case=case),
            )
            for case in cases_list
        ])
        scores_a: List[EvaluationResult] = [p[0] for p in pairs]
        scores_b: List[EvaluationResult] = [p[1] for p in pairs]

        per_dimension: Dict[EvalDimension, ABStatistics] = {}

        for dim in EvalDimension:
            series_a = [s.scores[dim].score for s in scores_a if dim in s.scores]
            series_b = [s.scores[dim].score for s in scores_b if dim in s.scores]
            if not series_a or not series_b:
                continue

            mean_a = statistics.mean(series_a)
            mean_b = statistics.mean(series_b)
            delta = mean_b - mean_a
            effect = self._compute_effect_size(series_a, series_b)

            per_dimension[dim] = ABStatistics(
                mean_a=mean_a,
                mean_b=mean_b,
                delta=delta,
                effect_size=effect,
            )

        result = ABComparisonResult(per_dimension=per_dimension)

        winner = result.best_prompt()
        overall_effect = (
            statistics.mean(s.effect_size for s in per_dimension.values())
            if per_dimension
            else 0.0
        )
        logger.info(
            "ab_comparison_complete",
            winner=winner,
            effect_size=round(overall_effect, 4),
            dimensions=len(per_dimension),
            cases=len(cases_list),
        )

        return result

