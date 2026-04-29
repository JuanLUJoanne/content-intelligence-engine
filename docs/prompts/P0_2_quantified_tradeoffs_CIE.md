# P0-2: Add Quantified Tradeoff Benchmarks (content-intelligence-engine)

## Context
The project has good qualitative design decisions in README but lacks quantified evidence. Tier 1 interviewers want to see "you proved this is better" not "you think this is better". We need to add benchmark scripts that measure the actual tradeoff numbers.

## What to do

### Step 1: Create benchmark framework
Create `scripts/benchmark_tradeoffs.py` that runs 3 key benchmarks using DummyLLM (free, deterministic, CI-safe):

**Benchmark 1: Heuristic routing vs single-tier**

Compare: route all to FLASH vs route all to PREMIUM vs heuristic routing.

Read `src/gateway/router.py` to understand the routing logic. The benchmark should:
1. Take the 50 eval records from `eval_data/product_metadata_50.jsonl` (from P0-1)
2. Run all 50 through the pipeline 3 times:
   - Force all to FLASH tier (cheapest)
   - Force all to PREMIUM tier (most expensive)
   - Use heuristic routing (actual behavior)
3. Measure per run: total_cost, avg_latency_ms, schema_compliance_rate, eval_score_avg
4. Output comparison table

Expected output format:
```json
{
  "benchmark": "routing_strategy",
  "variants": {
    "all_flash": {"cost": 0.XX, "latency_ms": XX, "compliance": 0.XX, "quality": 0.XX},
    "all_premium": {"cost": 0.XX, "latency_ms": XX, "compliance": 0.XX, "quality": 0.XX},
    "heuristic": {"cost": 0.XX, "latency_ms": XX, "compliance": 0.XX, "quality": 0.XX}
  },
  "conclusion": "Heuristic routing achieves X% of premium quality at Y% of the cost"
}
```

**Benchmark 2: Error-feedback retry vs blind retry**

Read `src/pipeline/graph.py` to understand how `_build_retry_prompt` works. The benchmark should:
1. Create 20 deliberately problematic inputs that will fail validation (e.g., ambiguous categories, missing price signals)
2. Run each input through 2 pipeline variants:
   - "blind_retry": on validation failure, retry with the SAME original prompt (no error feedback)
   - "error_feedback": on validation failure, retry with the Pydantic error included in prompt (current behavior)
3. Measure: fix_rate_attempt_1, fix_rate_attempt_2, fix_rate_attempt_3, total_fix_rate, avg_retries_to_fix

To implement "blind retry" variant: temporarily monkey-patch or parameterize the retry prompt builder to NOT include the error message.

Expected output:
```json
{
  "benchmark": "retry_strategy",
  "sample_size": 20,
  "variants": {
    "blind_retry": {"fix_rate": 0.XX, "avg_retries": X.X},
    "error_feedback": {"fix_rate": 0.XX, "avg_retries": X.X}
  },
  "delta": {
    "fix_rate_improvement": "+X.X pp",
    "retry_reduction": "-X.X retries avg"
  }
}
```

**Benchmark 3: LLM-as-Judge agreement with ground truth**

Read `src/eval/judge.py` to understand the judge scoring. The benchmark should:
1. Take 10 records from the eval dataset where we have `expected_output`
2. Run them through the pipeline to get `actual_output`
3. Score with LLM-as-Judge → get per-dimension scores
4. Compute "ground truth" scores by comparing actual vs expected (exact match for enums, fuzzy match for text fields)
5. Calculate correlation between judge scores and ground truth scores

Expected output:
```json
{
  "benchmark": "judge_agreement",
  "sample_size": 10,
  "agreement_rate": 0.XX,
  "per_dimension_correlation": {
    "factual_accuracy": 0.XX,
    "schema_compliance": 0.XX,
    "hallucination": 0.XX,
    "semantic_consistency": 0.XX,
    "relevance": 0.XX
  },
  "note": "Judge bias calibration: X/10 records where judge score diverges >0.2 from ground truth"
}
```

### Step 2: Run benchmarks
```bash
cd /Users/joelle/Projects/side-projects/content-intelligence-engine
source venv/bin/activate
python scripts/benchmark_tradeoffs.py
```

Write results to `eval_results/benchmark_tradeoffs.json`.

### Step 3: Add benchmark results to README

Add a new section after "Eval Results":

```markdown
## Quantified Design Tradeoffs

All benchmarks run on 50-record eval dataset. Reproducible via `python scripts/benchmark_tradeoffs.py`.

### Routing Strategy Impact

| Strategy | Cost | Latency | Compliance | Quality |
|----------|------|---------|------------|---------|
| All FLASH | $X.XX | Xms | X% | X.XX |
| All PREMIUM | $X.XX | Xms | X% | X.XX |
| Heuristic | $X.XX | Xms | X% | X.XX |

**Conclusion:** Heuristic routing achieves X% of premium quality at X% of the cost.

### Error-Feedback Retry vs Blind Retry

| Strategy | Fix Rate | Avg Retries |
|----------|----------|-------------|
| Blind retry | X% | X.X |
| Error feedback | X% | X.X |

**Conclusion:** Error feedback improves fix rate by +X pp and reduces avg retries by X.X.

### LLM-as-Judge Calibration

Judge agreement with ground truth labels: **X%** (N=10).
Per-dimension correlation: factual_accuracy X.XX, schema_compliance X.XX, hallucination X.XX.
```

Fill in actual numbers from the benchmark run.

### Step 4: Run all tests
```bash
pytest tests/ -v
```
Ensure no regressions.

### IMPORTANT NOTES
- Use DummyLLM for all benchmarks — this is about measuring the SYSTEM behavior, not the LLM quality
- The numbers from DummyLLM are still meaningful: they show the pipeline's error handling, retry logic, and routing decisions work correctly
- Read the actual source files before writing code — match real constructor signatures and state keys
- Keep the benchmark script under 200 lines. Simple, readable, no over-engineering.
