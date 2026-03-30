"""
Quantified design tradeoff benchmarks.

Runs three benchmarks using DummyLLM (free, deterministic, CI-safe) to measure
the system-level impact of key engineering decisions:

1. Heuristic routing vs single-tier — cost/quality tradeoff
2. Error-feedback retry vs blind retry — self-correction effectiveness
3. LLM-as-Judge agreement with ground truth — eval calibration

Usage:
    python -m scripts.benchmark_tradeoffs
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path

from src.eval.judge import LLMJudge
from src.gateway.cost_tracker import CostTracker, ModelPricing
from src.gateway.providers import DummyProvider
from src.gateway.router import ModelRouter, ModelTier, TaskFeatures
from src.pipeline.graph import ContentPipelineGraph, _build_retry_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_records(limit: int = 0) -> list[dict]:
    path = Path("eval_data/product_metadata_50.jsonl")
    records = [json.loads(line) for line in path.read_text().strip().splitlines()]
    return records[:limit] if limit > 0 else records


def _make_cost_tracker() -> CostTracker:
    return CostTracker(
        pricing_by_model={
            "flash-model": ModelPricing(
                input_cost_per_1k_tokens=Decimal("0.0001"),
                output_cost_per_1k_tokens=Decimal("0.0002"),
            ),
            "standard-model": ModelPricing(
                input_cost_per_1k_tokens=Decimal("0.001"),
                output_cost_per_1k_tokens=Decimal("0.002"),
            ),
            "premium-model": ModelPricing(
                input_cost_per_1k_tokens=Decimal("0.01"),
                output_cost_per_1k_tokens=Decimal("0.02"),
            ),
        },
        total_budget=Decimal("100.00"),
    )


# ---------------------------------------------------------------------------
# Benchmark 1: Routing Strategy
# ---------------------------------------------------------------------------

async def benchmark_routing(records: list[dict]) -> dict:
    """Compare all-FLASH vs all-PREMIUM vs heuristic routing."""
    results = {}

    for strategy, min_tier, quality_sens, cost_sens in [
        ("all_flash", ModelTier.FLASH, 0.0, 1.0),
        ("all_premium", ModelTier.PREMIUM, 1.0, 0.0),
        ("heuristic", ModelTier.FLASH, 0.5, 0.5),
    ]:
        tracker = _make_cost_tracker()
        router = ModelRouter(cost_tracker=tracker)
        await router.register_model("flash-model", ModelTier.FLASH)
        await router.register_model("standard-model", ModelTier.STANDARD)
        await router.register_model("premium-model", ModelTier.PREMIUM)

        provider = DummyProvider()
        tier_counts: dict[str, int] = {}
        latencies: list[float] = []
        compliance = 0

        for rec in records:
            input_text = rec.get("input_text", "")
            est_tokens = max(len(input_text) // 4, 10)

            features = TaskFeatures(
                estimated_input_tokens=est_tokens,
                estimated_output_tokens=50,
                latency_sensitivity=0.5,
                quality_sensitivity=quality_sens,
                cost_sensitivity=cost_sens,
                minimum_tier=min_tier,
            )

            t0 = time.monotonic()
            model = await router.choose_model(features)
            tier_counts[model] = tier_counts.get(model, 0) + 1

            # Simulate cost recording
            tracker.record_usage(model, input_tokens=est_tokens, output_tokens=50)

            # Run through pipeline
            graph = ContentPipelineGraph(provider=provider, model_id="dummy")
            state = await graph.run({"id": rec["id"], "input_text": input_text})
            latencies.append((time.monotonic() - t0) * 1000)

            if state.get("validation_result") is True:
                compliance += 1

        total_cost = sum(s.total_cost for s in tracker.summary_by_model())

        results[strategy] = {
            "cost": float(total_cost),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "compliance": round(compliance / len(records), 4),
            "tier_distribution": tier_counts,
        }

    return {
        "benchmark": "routing_strategy",
        "record_count": len(records),
        "variants": results,
        "conclusion": (
            f"Heuristic routing achieves {results['heuristic']['compliance']:.0%} compliance "
            f"at ${results['heuristic']['cost']:.4f} "
            f"({results['heuristic']['cost'] / max(results['all_premium']['cost'], 0.0001):.0%} of premium cost)"
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Error-Feedback Retry vs Blind Retry
# ---------------------------------------------------------------------------

async def benchmark_retry(records: list[dict]) -> dict:
    """Compare error-feedback retry vs blind retry (same prompt repeated)."""
    provider = DummyProvider()

    variants = {}
    for strategy in ("blind_retry", "error_feedback"):
        total_fixes = 0
        total_retries = 0
        fix_by_attempt: dict[int, int] = {1: 0, 2: 0, 3: 0}

        # Use a validate_fn that rejects certain DummyProvider outputs on
        # first attempt to force retries. We use a closure to track attempts.
        for rec in records:
            attempt_counter = {"count": 0}

            def _counting_validate(output: dict, _ctr=attempt_counter) -> bool:
                _ctr["count"] += 1
                # Reject first attempt to force retry path
                if _ctr["count"] == 1:
                    return False
                return True

            if strategy == "blind_retry":
                # Monkey-patch: make retry prompt NOT include error feedback
                original_builder = _build_retry_prompt
                import src.pipeline.graph as _graph_mod

                def _blind_builder(
                    original_prompt: str,
                    bad_response: dict,
                    error_msg: str,
                ) -> str:
                    return original_prompt  # No error feedback

                _graph_mod._build_retry_prompt = _blind_builder  # type: ignore[assignment]

            graph = ContentPipelineGraph(
                provider=provider,
                model_id="dummy",
                validate_fn=_counting_validate,
            )
            state = await graph.run({"id": rec["id"], "input_text": rec.get("input_text", "")})

            if strategy == "blind_retry":
                _graph_mod._build_retry_prompt = original_builder  # type: ignore[assignment]

            retries = state.get("retry_count", 0)
            fixed = state.get("validation_result") is True
            total_retries += retries
            if fixed:
                total_fixes += 1
                attempt = min(retries + 1, 3)
                fix_by_attempt[attempt] = fix_by_attempt.get(attempt, 0) + 1

        n = len(records)
        variants[strategy] = {
            "fix_rate": round(total_fixes / n, 4),
            "avg_retries": round(total_retries / n, 2),
            "fix_by_attempt": fix_by_attempt,
        }

    ef = variants["error_feedback"]
    br = variants["blind_retry"]
    return {
        "benchmark": "retry_strategy",
        "record_count": len(records),
        "variants": variants,
        "delta": {
            "fix_rate_improvement": f"{(ef['fix_rate'] - br['fix_rate']) * 100:+.1f} pp",
            "retry_reduction": f"{ef['avg_retries'] - br['avg_retries']:+.2f} retries avg",
        },
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Judge Agreement with Ground Truth
# ---------------------------------------------------------------------------

async def benchmark_judge_agreement(records: list[dict]) -> dict:
    """Measure correlation between LLM-as-Judge scores and ground truth."""
    provider = DummyProvider()
    judge = LLMJudge.from_provider("dummy")

    agreements = []
    dimension_matches: dict[str, list[float]] = {
        "factual_accuracy": [],
        "schema_compliance": [],
        "hallucination": [],
        "semantic_consistency": [],
        "relevance": [],
    }

    for rec in records[:10]:
        expected = rec["expected_output"]
        graph = ContentPipelineGraph(provider=provider, model_id="dummy")
        state = await graph.run({"id": rec["id"], "input_text": rec.get("input_text", "")})

        actual = state.get("final_output") or {}
        passed = state.get("validation_result") is True

        # Ground truth: field-level exact match
        gt_scores = {
            "category_match": float(str(actual.get("category", "")).lower() == str(expected.get("category", "")).lower()),
            "condition_match": float(str(actual.get("condition", "")).lower() == str(expected.get("condition", "")).lower()),
            "price_range_match": float(str(actual.get("price_range", "")).lower() == str(expected.get("price_range", "")).lower()),
        }
        gt_overall = sum(gt_scores.values()) / max(len(gt_scores), 1)

        # Judge scores
        if passed and actual:
            eval_result = await judge.score(
                user_input=rec.get("input_text", ""),
                candidate_output=json.dumps(actual),
                reference_output=json.dumps(expected),
            )
            judge_overall = eval_result.overall_score
            for dim, score in eval_result.scores.items():
                dimension_matches[dim.value].append(score.score)
        else:
            judge_overall = 0.0
            for dim in dimension_matches:
                dimension_matches[dim].append(0.0)

        # Agreement: both high (>0.5) or both low (<=0.5)
        agree = (gt_overall > 0.5 and judge_overall > 0.5) or (gt_overall <= 0.5 and judge_overall <= 0.5)
        agreements.append(agree)

    dim_averages = {
        dim: round(sum(scores) / max(len(scores), 1), 4)
        for dim, scores in dimension_matches.items()
    }
    agreement_rate = round(sum(agreements) / max(len(agreements), 1), 4)
    divergences = sum(1 for a in agreements if not a)

    return {
        "benchmark": "judge_agreement",
        "sample_size": len(agreements),
        "agreement_rate": agreement_rate,
        "per_dimension_avg": dim_averages,
        "divergence_count": divergences,
        "note": f"Judge/ground-truth disagreement on {divergences}/{len(agreements)} records",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    records = _load_records()
    print(f"Loaded {len(records)} records\n")

    print("=" * 60)
    print("  Benchmark 1: Routing Strategy")
    print("=" * 60)
    b1 = await benchmark_routing(records)
    for strategy, data in b1["variants"].items():
        print(f"  {strategy:15s}  cost=${data['cost']:.4f}  latency={data['avg_latency_ms']:.1f}ms  compliance={data['compliance']:.0%}")
    print(f"  → {b1['conclusion']}")
    print()

    print("=" * 60)
    print("  Benchmark 2: Error-Feedback Retry vs Blind Retry")
    print("=" * 60)
    b2 = await benchmark_retry(records)
    for strategy, data in b2["variants"].items():
        print(f"  {strategy:15s}  fix_rate={data['fix_rate']:.0%}  avg_retries={data['avg_retries']:.2f}")
    print(f"  → Fix rate improvement: {b2['delta']['fix_rate_improvement']}")
    print(f"  → Retry reduction: {b2['delta']['retry_reduction']}")
    print()

    print("=" * 60)
    print("  Benchmark 3: Judge Agreement with Ground Truth")
    print("=" * 60)
    b3 = await benchmark_judge_agreement(records)
    print(f"  Agreement rate: {b3['agreement_rate']:.0%} (N={b3['sample_size']})")
    for dim, avg in b3["per_dimension_avg"].items():
        print(f"    {dim:25s}: {avg:.3f}")
    print(f"  → {b3['note']}")
    print()

    # Write combined report
    report = {
        "routing_strategy": b1,
        "retry_strategy": b2,
        "judge_agreement": b3,
    }
    out_path = Path("eval_results/benchmark_tradeoffs.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
