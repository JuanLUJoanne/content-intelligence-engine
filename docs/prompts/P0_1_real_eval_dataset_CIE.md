# P0-1: Add Real Eval Dataset + Run Real Numbers (content-intelligence-engine)

## Context
This project has a 7-node LLM pipeline in `src/pipeline/graph.py` that processes product metadata through: sanitize → cache → route → LLM call → validate → eval → store. Currently all metrics in README (schema compliance 98.5%, hallucination 3.1%, cost $0.0011/record) come from DummyLLM + mock evaluator. We need real numbers from a real LLM to be credible.

## What to do

### Step 1: Create eval dataset
Create `eval_data/product_metadata_50.jsonl` with 50 hand-crafted product metadata records. Each line is a JSON object:

```json
{
  "id": "eval-001",
  "input_text": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones, Black, Bluetooth, 30hr battery",
  "expected_output": {
    "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
    "category": "electronics",
    "condition": "new",
    "price_range": "mid_range",
    "tags": ["sony", "headphones", "noise_cancelling", "bluetooth", "wireless"],
    "description": "Wireless noise cancelling headphones with Bluetooth connectivity and 30-hour battery life.",
    "language": "en"
  }
}
```

Requirements:
- Cover ALL category enum values from `src/schemas/metadata.py`: `electronics`, `clothing`, `home_garden`, `books`, `sports`, `toys`
- Cover ALL condition enum values: `new`, `like_new`, `good`, `fair`
- Cover ALL price_range enum values: `budget`, `mid_range`, `premium`, `luxury`, `unpriced`
- Include edge cases: very short descriptions (5 words), very long descriptions (200+ words), ambiguous categories, multi-language product names, products with no clear price signal
- 50 records total, diverse and realistic

### Step 2: Create eval runner script
Create `scripts/run_eval.py`:

```python
"""
Run real LLM evaluation on the 50-record dataset.

Usage:
    # With real Gemini API (costs ~$0.02):
    GOOGLE_API_KEY=xxx python scripts/run_eval.py --provider gemini

    # With DummyLLM (free, for CI):
    python scripts/run_eval.py --provider dummy
"""
import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Import the pipeline
from src.pipeline.graph import ContentPipelineGraph
from src.gateway.providers import DummyProvider
from src.eval.judge import LLMJudge  # or whatever the judge class is called


async def main(provider_name: str):
    dataset_path = Path("eval_data/product_metadata_50.jsonl")
    records = [json.loads(line) for line in dataset_path.read_text().strip().split("\n")]

    # Build pipeline with chosen provider
    if provider_name == "dummy":
        # Use DummyProvider for CI
        provider = DummyProvider()
    elif provider_name == "gemini":
        from src.gateway.providers import GeminiProvider
        provider = GeminiProvider(model_id="gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    graph = ContentPipelineGraph(provider=provider)

    results = []
    for record in records:
        state = await graph.run(record)
        results.append({
            "id": record["id"],
            "input": record["input_text"],
            "expected": record["expected_output"],
            "actual": state.get("final_output"),
            "validation_passed": state.get("validation_result") is True,
            "retry_count": state.get("retry_count", 0),
            "failure_type": state.get("failure_type"),
            "eval_score": state.get("eval_score"),
            "eval_dimensions": state.get("eval_dimensions"),
            "model_used": state.get("model_id"),
            "cost": str(state.get("cost", 0)),
        })

    # Compute aggregate metrics
    total = len(results)
    valid = sum(1 for r in results if r["validation_passed"])
    retried = sum(1 for r in results if r["retry_count"] > 0)
    failed = sum(1 for r in results if r["failure_type"])

    schema_compliance = valid / total if total else 0
    avg_retries = sum(r["retry_count"] for r in results) / total if total else 0
    total_cost = sum(float(r["cost"]) for r in results)

    # Eval dimension averages (if judge ran)
    dim_scores = {}
    scored = [r for r in results if r["eval_dimensions"]]
    if scored:
        for dim in scored[0]["eval_dimensions"]:
            dim_scores[dim] = sum(
                r["eval_dimensions"][dim] for r in scored
            ) / len(scored)

    report = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "dataset": str(dataset_path),
        "record_count": total,
        "schema_compliance": round(schema_compliance, 4),
        "retry_rate": round(retried / total, 4) if total else 0,
        "structural_failure_rate": round(
            sum(1 for r in results if r["failure_type"] == "structural_failure") / total, 4
        ) if total else 0,
        "avg_retries_per_record": round(avg_retries, 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_record": round(total_cost / total, 6) if total else 0,
        "eval_dimension_averages": dim_scores,
        "per_record_results": results,
    }

    # Write report
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{provider_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nEval report written to: {out_path}")
    print(f"Schema compliance: {schema_compliance:.1%}")
    print(f"Retry rate: {retried}/{total}")
    print(f"Total cost: ${total_cost:.4f}")
    if dim_scores:
        for dim, score in dim_scores.items():
            print(f"  {dim}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="dummy", choices=["dummy", "gemini"])
    args = parser.parse_args()
    asyncio.run(main(args.provider))
```

IMPORTANT: Adapt the script to match the actual constructor signatures and state keys used in the codebase. Read `src/pipeline/graph.py`, `src/eval/judge.py`, and `src/gateway/providers.py` first to understand the exact API. The code above is a template — adjust imports, constructor args, and state field names to match reality.

### Step 3: Run with DummyLLM first
```bash
cd /Users/joelle/Projects/side-projects/content-intelligence-engine
source venv/bin/activate
python scripts/run_eval.py --provider dummy
```

Ensure it produces a valid `eval_results/eval_dummy_*.json` file. Fix any import or API mismatch errors.

### Step 4: Add eval dataset to .gitignore exception
Ensure `eval_data/` and `eval_results/` are NOT in .gitignore — these should be committed as evidence.

### Step 5: Update README metrics section
Replace the current metrics table with a note:

```markdown
## Eval Results

Measured on 50 manually-curated product metadata records (`eval_data/product_metadata_50.jsonl`).

| Metric | DummyLLM (CI) | Gemini 2.0 Flash | Notes |
|--------|--------------|-------------------|-------|
| Schema Compliance | X% | (run with API key) | After error-feedback retries |
| Avg Retries / Record | X | | Lower = better prompt |
| Structural Failure Rate | X% | | Indicates prompt regression |
| Avg Cost / Record | $0 | $X.XXXX | Flash tier |
| Factual Accuracy | X.XX | | LLM-as-Judge |
| Hallucination Rate | X.XX | | LLM-as-Judge |

_Run `python scripts/run_eval.py --provider gemini` to reproduce._
```

Fill in the DummyLLM column from the actual run. Leave Gemini column for user to fill after running with API key.

### Step 6: Run all existing tests
```bash
pytest tests/ -v
```
Ensure no regressions. All 251+ tests should still pass.
