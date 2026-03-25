"""
Run evaluation on the 50-record product metadata dataset.

Executes the full ContentPipelineGraph on each record, then runs LLM-as-Judge
scoring against the expected output. Produces a JSON report in eval_results/.

Usage:
    # DummyLLM (free, works without API keys — used by CI):
    python scripts/run_eval.py --provider dummy

    # Gemini 2.0 Flash (~$0.02 for 50 records):
    GOOGLE_API_KEY=xxx python scripts/run_eval.py --provider gemini

    # With judge scoring enabled (adds 5 LLM calls per record):
    python scripts/run_eval.py --provider dummy --judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env file from project root (parent of scripts/)
# Works regardless of which directory the script is run from.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

# Pipeline imports
from src.eval.judge import EvalDimension, LLMJudge
from src.gateway.providers import DummyProvider, ProviderFactory
from src.pipeline.graph import ContentPipelineGraph


# ---------------------------------------------------------------------------
# Ground-truth comparison helpers
# ---------------------------------------------------------------------------


def _field_match(actual: dict, expected: dict, field: str) -> bool:
    """Case-insensitive equality check for a single field."""
    a = str(actual.get(field, "")).strip().lower()
    e = str(expected.get(field, "")).strip().lower()
    return a == e


def _tag_overlap(actual_tags: list[str], expected_tags: list[str]) -> float:
    """Fraction of expected tags present in actual tags (recall)."""
    if not expected_tags:
        return 1.0
    actual_set = {t.strip().lower().replace(" ", "_") for t in actual_tags}
    expected_set = {t.strip().lower().replace(" ", "_") for t in expected_tags}
    return len(actual_set & expected_set) / len(expected_set)


def _ground_truth_score(actual: dict | None, expected: dict) -> dict[str, float]:
    """
    Compute deterministic ground-truth scores by comparing actual vs expected.

    These are structural checks (correct enum values, tag overlap) — not LLM-based.
    Useful even with DummyLLM because they measure pipeline routing accuracy.
    """
    if actual is None:
        return {
            "category_match": 0.0,
            "condition_match": 0.0,
            "price_range_match": 0.0,
            "tag_recall": 0.0,
            "overall_structural": 0.0,
        }

    cat = float(_field_match(actual, expected, "category"))
    cond = float(_field_match(actual, expected, "condition"))
    price = float(_field_match(actual, expected, "price_range"))
    tags = _tag_overlap(actual.get("tags", []), expected.get("tags", []))
    overall = (cat + cond + price + tags) / 4

    return {
        "category_match": cat,
        "condition_match": cond,
        "price_range_match": price,
        "tag_recall": round(tags, 4),
        "overall_structural": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


async def run_eval(provider_name: str, use_judge: bool, delay: float = 0.0, limit: int = 0) -> None:
    dataset_path = Path("eval_data/product_metadata_50.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run from project root.")

    records = [json.loads(line) for line in dataset_path.read_text().strip().splitlines()]
    if limit > 0:
        records = records[:limit]
    print(f"Loaded {len(records)} records from {dataset_path}")

    # Build pipeline provider
    if provider_name == "dummy":
        provider = DummyProvider()
        judge = LLMJudge.from_provider("dummy") if use_judge else None
    elif provider_name == "gemini":
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY env var required for --provider gemini")
        from src.gateway.providers import GeminiProvider
        provider = GeminiProvider(api_key=api_key)
        judge = LLMJudge.from_provider("gemini-3.1-flash-lite-preview") if use_judge else None
    else:
        raise ValueError(f"Unknown provider: {provider_name!r}. Use 'dummy' or 'gemini'.")

    graph = ContentPipelineGraph(provider=provider, model_id="dummy" if provider_name == "dummy" else "gemini-3.1-flash-lite-preview")

    results = []
    t_start = time.monotonic()

    for i, record in enumerate(records, 1):
        record_id = record["id"]
        input_text = record["input_text"]
        expected = record["expected_output"]

        t0 = time.monotonic()
        state = await graph.run({"id": record_id, "input_text": input_text})
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        actual_output = state.get("final_output")

        # Structural ground-truth scores (deterministic, no LLM)
        gt_scores = _ground_truth_score(actual_output, expected)

        # LLM-as-Judge scores (optional, requires provider)
        judge_scores: dict[str, float] = {}
        judge_overall: float | None = None
        if judge and actual_output and state.get("validation_result") is True:
            eval_result = await judge.score(
                user_input=input_text,
                candidate_output=json.dumps(actual_output),
                reference_output=json.dumps(expected),
                inter_call_delay=delay,
            )
            judge_scores = {
                dim.value: score.score
                for dim, score in eval_result.scores.items()
            }
            judge_overall = round(eval_result.overall_score, 4)

        results.append({
            "id": record_id,
            "input": input_text,
            "expected_category": expected.get("category"),
            "expected_condition": expected.get("condition"),
            "expected_price_range": expected.get("price_range"),
            "actual_output": actual_output,
            "validation_passed": state.get("validation_result") is True,
            "retry_count": state.get("retry_count", 0),
            "failure_type": state.get("failure_type"),
            "sent_to_review": state.get("sent_to_review", False),
            "sent_to_engineering": state.get("sent_to_engineering", False),
            "sent_to_dlq": state.get("sent_to_dlq", False),
            "eval_score": state.get("eval_score"),
            "latency_ms": latency_ms,
            "ground_truth_scores": gt_scores,
            "judge_scores": judge_scores if judge_scores else None,
            "judge_overall": judge_overall,
        })

        status = "✓" if state.get("validation_result") is True else "✗"
        extra = f"retry={state['retry_count']}" if state.get("retry_count", 0) > 0 else ""
        print(f"  [{i:02d}/{len(records)}] {status} {record_id}  structural={gt_scores['overall_structural']:.2f}  {extra}")

        if delay > 0 and i < len(records):
            await asyncio.sleep(delay)

    total_elapsed = round(time.monotonic() - t_start, 1)

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    total = len(results)
    validated = [r for r in results if r["validation_passed"]]
    retried = [r for r in results if r["retry_count"] > 0]
    structural_failures = [r for r in results if r["sent_to_engineering"]]
    review_routes = [r for r in results if r["sent_to_review"]]

    schema_compliance = len(validated) / total if total else 0.0
    avg_retries = sum(r["retry_count"] for r in results) / total if total else 0.0

    # Ground-truth averages (all records)
    gt_keys = ["category_match", "condition_match", "price_range_match", "tag_recall", "overall_structural"]
    gt_averages = {
        k: round(sum(r["ground_truth_scores"][k] for r in results) / total, 4)
        for k in gt_keys
    }

    # Judge averages (only records that were scored)
    judge_scored = [r for r in results if r["judge_scores"]]
    judge_averages: dict[str, float] = {}
    if judge_scored:
        for dim in EvalDimension:
            vals = [r["judge_scores"][dim.value] for r in judge_scored if r["judge_scores"] and dim.value in r["judge_scores"]]
            if vals:
                judge_averages[dim.value] = round(sum(vals) / len(vals), 4)

    report = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "judge_enabled": use_judge,
        "dataset": str(dataset_path),
        "record_count": total,
        "total_elapsed_s": total_elapsed,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 1) if total else 0,
        "schema_compliance": round(schema_compliance, 4),
        "retry_rate": round(len(retried) / total, 4) if total else 0,
        "avg_retries_per_record": round(avg_retries, 3),
        "structural_failure_rate": round(len(structural_failures) / total, 4) if total else 0,
        "review_route_rate": round(len(review_routes) / total, 4) if total else 0,
        "dlq_rate": round(sum(1 for r in results if r["sent_to_dlq"]) / total, 4) if total else 0,
        "ground_truth_averages": gt_averages,
        "judge_averages": judge_averages if judge_averages else None,
        "per_record_results": results,
    }

    # ---------------------------------------------------------------------------
    # Write report
    # ---------------------------------------------------------------------------
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{provider_name}_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Eval complete  ({total_elapsed}s)")
    print("=" * 60)
    print(f"  Provider          : {provider_name}")
    print(f"  Records           : {total}")
    dlq_routes = [r for r in results if r["sent_to_dlq"]]
    print(f"  Schema compliance : {schema_compliance:.1%}  ({len(validated)}/{total} passed validation)")
    print(f"  Retry rate        : {len(retried)}/{total}  (avg {avg_retries:.2f} retries/record)")
    print(f"  Structural failures: {len(structural_failures)}/{total}  (→ engineering queue)")
    print(f"  Review routes     : {len(review_routes)}/{total}  (→ human review)")
    print(f"  DLQ routes        : {len(dlq_routes)}/{total}  (API/network errors)")
    if dlq_routes:
        sample_err = results[0].get("actual_output") or "(no output — API error)"
        print(f"    └─ If all records hit DLQ: check API key and network access")
    print()
    print("  Ground-truth accuracy (vs expected_output):")
    for k, v in gt_averages.items():
        print(f"    {k:<25}: {v:.1%}")
    if judge_averages:
        print()
        print("  LLM-as-Judge scores:")
        for dim, score in judge_averages.items():
            print(f"    {dim:<25}: {score:.3f}")
    print()
    print(f"  Report: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate content pipeline on 50-record dataset")
    parser.add_argument(
        "--provider",
        default="dummy",
        choices=["dummy", "gemini"],
        help="LLM provider to use. 'dummy' works without API keys (default).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-Judge scoring (5 extra calls per record).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between records. Use 4+ for free-tier rate limits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run first N records (0 = all 50). Useful for quick tests.",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args.provider, args.judge, args.delay, args.limit))
