"""Tests for PromptComparator A/B evaluation and ABComparisonResult."""

from __future__ import annotations

import statistics
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.ab_prompt import ABComparisonResult, ABStatistics, PromptComparator, PromptTestCase
from src.eval.judge import DimensionScore, EvalDimension, EvaluationResult, LLMJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(score: float) -> EvaluationResult:
    """Return an EvaluationResult with all five dimensions at the same score."""
    return EvaluationResult(
        scores={
            dim: DimensionScore(score=score, reasoning="test")
            for dim in EvalDimension
        }
    )


def _make_judge(side_effect: list[EvaluationResult]) -> LLMJudge:
    """Return a mock LLMJudge whose score() calls are consumed from side_effect."""
    mock = MagicMock(spec=LLMJudge)
    mock.score = AsyncMock(side_effect=side_effect)
    return mock


# ---------------------------------------------------------------------------
# compare() — winner selection
# ---------------------------------------------------------------------------


class TestCompareWinner:
    @pytest.mark.asyncio
    async def test_returns_a_when_prompt_a_clearly_better(self):
        # Prompt A scores 0.9, prompt B scores 0.5 → delta = 0.4 > 0.02 → "A"
        cases = [PromptTestCase(user_input="x")]
        judge = _make_judge([_make_eval_result(0.9), _make_eval_result(0.5)])
        comparator = PromptComparator(judge=judge)

        result = await comparator.compare(
            prompt_a="Prompt A: {input}",
            prompt_b="Prompt B: {input}",
            cases=cases,
        )

        assert result.best_prompt() == "A"

    @pytest.mark.asyncio
    async def test_returns_b_when_prompt_b_clearly_better(self):
        cases = [PromptTestCase(user_input="x")]
        judge = _make_judge([_make_eval_result(0.4), _make_eval_result(0.9)])
        comparator = PromptComparator(judge=judge)

        result = await comparator.compare(
            prompt_a="A: {input}",
            prompt_b="B: {input}",
            cases=cases,
        )

        assert result.best_prompt() == "B"

    @pytest.mark.asyncio
    async def test_returns_none_when_scores_identical(self):
        cases = [PromptTestCase(user_input="x")]
        judge = _make_judge([_make_eval_result(0.8), _make_eval_result(0.8)])
        comparator = PromptComparator(judge=judge)

        result = await comparator.compare(
            prompt_a="A: {input}",
            prompt_b="B: {input}",
            cases=cases,
        )

        assert result.best_prompt() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_delta_below_threshold(self):
        # Delta of 0.01 < 0.02 threshold → None
        cases = [PromptTestCase(user_input="x")]
        judge = _make_judge([_make_eval_result(0.81), _make_eval_result(0.80)])
        comparator = PromptComparator(judge=judge)

        result = await comparator.compare(
            prompt_a="A: {input}",
            prompt_b="B: {input}",
            cases=cases,
        )

        assert result.best_prompt() is None

    @pytest.mark.asyncio
    async def test_multi_case_aggregates_correctly(self):
        # 2 cases: A=[0.9, 0.8]=avg 0.85, B=[0.5, 0.6]=avg 0.55 → "A"
        cases = [PromptTestCase(user_input="x"), PromptTestCase(user_input="y")]
        judge = _make_judge([
            _make_eval_result(0.9),  # A case 1
            _make_eval_result(0.5),  # B case 1
            _make_eval_result(0.8),  # A case 2
            _make_eval_result(0.6),  # B case 2
        ])
        comparator = PromptComparator(judge=judge)

        result = await comparator.compare(
            prompt_a="A: {input}",
            prompt_b="B: {input}",
            cases=cases,
        )

        assert result.best_prompt() == "A"


# ---------------------------------------------------------------------------
# Effect size calculation
# ---------------------------------------------------------------------------


class TestEffectSize:
    def test_effect_size_zero_when_identical(self):
        a = [0.8, 0.8]
        b = [0.8, 0.8]
        effect = PromptComparator._compute_effect_size(a, b)
        assert effect == pytest.approx(0.0)

    def test_effect_size_negative_when_b_lower(self):
        # a=[0.8, 1.0], b=[0.0, 0.2]
        # mean_a=0.9, mean_b=0.1
        # pvariance(a)=0.01, pvariance(b)=0.01 → pooled_std=0.1
        # effect = (0.1 - 0.9) / 0.1 = -8.0
        a = [0.8, 1.0]
        b = [0.0, 0.2]
        effect = PromptComparator._compute_effect_size(a, b)
        assert effect == pytest.approx(-8.0)

    def test_effect_size_positive_when_b_higher(self):
        a = [0.0, 0.2]
        b = [0.8, 1.0]
        effect = PromptComparator._compute_effect_size(a, b)
        assert effect == pytest.approx(8.0)

    def test_effect_size_zero_when_empty(self):
        assert PromptComparator._compute_effect_size([], [0.8]) == 0.0
        assert PromptComparator._compute_effect_size([0.8], []) == 0.0

    def test_effect_size_known_cohens_d(self):
        # a=[0.6, 0.8], b=[0.2, 0.4]
        # mean_a=0.7, mean_b=0.3
        # pvariance(a)=0.01, pvariance(b)=0.01 → pooled_std=0.1
        # effect = (0.3 - 0.7) / 0.1 = -4.0
        a = [0.6, 0.8]
        b = [0.2, 0.4]
        effect = PromptComparator._compute_effect_size(a, b)
        assert effect == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# ABComparisonResult.best_prompt() threshold
# ---------------------------------------------------------------------------


class TestBestPrompt:
    def _make_result(self, mean_a: float, mean_b: float) -> ABComparisonResult:
        stats = ABStatistics(mean_a=mean_a, mean_b=mean_b, delta=mean_b - mean_a, effect_size=0.0)
        return ABComparisonResult(per_dimension={EvalDimension.RELEVANCE: stats})

    def test_none_when_no_dimensions(self):
        assert ABComparisonResult(per_dimension={}).best_prompt() is None

    def test_a_wins_when_delta_large_enough(self):
        assert self._make_result(0.9, 0.5).best_prompt() == "A"

    def test_b_wins_when_delta_large_enough(self):
        assert self._make_result(0.5, 0.9).best_prompt() == "B"

    def test_none_when_delta_exactly_at_threshold(self):
        # |0.8 - 0.82| = 0.02 → None (not strictly greater than 0.02)
        assert self._make_result(0.80, 0.82).best_prompt() is None

    def test_none_when_delta_just_below_threshold(self):
        assert self._make_result(0.80, 0.819).best_prompt() is None
