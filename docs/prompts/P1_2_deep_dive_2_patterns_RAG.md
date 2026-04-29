# P1-2: Deep Dive 2 Patterns in Agentic RAG (Reduce Signal Dilution)

## Context
Project at `/Users/joelle/Projects/side-projects/agentic-rag`. Currently showcases 10 agentic patterns but none are deep enough. The README lists all 10 equally, which dilutes the signal. We need to pick 2 patterns and add depth: benchmarks, failure modes, parameter sensitivity analysis.

The 2 patterns to deep-dive:
1. **Parallel Retrieval + RRF** (pattern #9) — `src/retrieval/parallel_retriever.py`
2. **Supervisor Multi-Agent Orchestration** (pattern #4) — `src/agents/supervisor.py`, `src/graph/multi_agent_workflow.py`

## What to do

### Step 1: RRF Parameter Sensitivity Analysis

Read `src/retrieval/parallel_retriever.py` to understand the RRF implementation. The RRF formula is: `score(doc) = Σ 1/(k + rank_i)` where k is typically 60.

Create `scripts/benchmark_rrf_sensitivity.py` that:

1. Uses the eval dataset from P0-3 (`eval_data/qa_100.jsonl`) — if it doesn't exist yet, create 20 test queries inline
2. Runs parallel retrieval with different k values: [1, 10, 20, 40, 60, 100, 200]
3. For each k value, measures:
   - MRR@5 (Mean Reciprocal Rank at top 5)
   - Rank correlation with each individual retriever (Kendall's tau or Spearman)
   - How often RRF changes the top-1 result vs the best single retriever

Output to `eval_results/rrf_sensitivity.json`:
```json
{
  "benchmark": "rrf_k_sensitivity",
  "k_values_tested": [1, 10, 20, 40, 60, 100, 200],
  "results": [
    {"k": 1, "mrr_at_5": 0.XX, "top1_change_rate": 0.XX, "note": "Almost purely rank-position weighted"},
    {"k": 60, "mrr_at_5": 0.XX, "top1_change_rate": 0.XX, "note": "Standard balanced"},
    {"k": 200, "mrr_at_5": 0.XX, "top1_change_rate": 0.XX, "note": "Near-uniform weighting"}
  ],
  "best_k": 60,
  "analysis": "k=60 maximizes MRR because... k<20 over-weights first retriever's top result, k>100 makes fusion nearly random..."
}
```

### Step 2: Supervisor Decision Quality Analysis

Read `src/agents/supervisor.py` and `src/graph/multi_agent_workflow.py` to understand how the supervisor routes queries.

Create `scripts/benchmark_supervisor.py` that:

1. Creates 50 test queries across difficulty levels:
   - 15 simple factual (should route to ResearchAgent directly)
   - 15 comparison (should route to AnalysisAgent)
   - 10 multi-hop (should route to ResearchAgent → AnalysisAgent)
   - 10 ambiguous (routing is uncertain — interesting to measure)

2. For each query, records:
   - Which agent(s) the supervisor chose
   - Whether the choice was "optimal" (based on our expected routing above)
   - Total steps executed
   - Whether the quality agent triggered a retry

3. Computes:
   - **Routing accuracy**: % of queries routed to the expected agent
   - **Misroute cost**: when supervisor misroutes, how many extra steps before correction?
   - **Overhead**: supervisor decision latency vs direct routing

Output to `eval_results/supervisor_analysis.json`:
```json
{
  "benchmark": "supervisor_routing_quality",
  "query_count": 50,
  "routing_accuracy": 0.XX,
  "by_difficulty": {
    "simple": {"accuracy": 0.XX, "avg_steps": X.X},
    "comparison": {"accuracy": 0.XX, "avg_steps": X.X},
    "multi_hop": {"accuracy": 0.XX, "avg_steps": X.X},
    "ambiguous": {"accuracy": 0.XX, "avg_steps": X.X}
  },
  "misroute_analysis": {
    "misroute_rate": 0.XX,
    "avg_extra_steps_on_misroute": X.X,
    "recovery_rate": 0.XX
  },
  "overhead": {
    "supervisor_decision_latency_ms": XX,
    "direct_routing_latency_ms": XX,
    "delta_ms": XX
  }
}
```

### Step 3: Add failure mode documentation

Create `docs/failure-modes.md` (in agentic-rag project):

```markdown
# Failure Modes & Recovery

## Parallel Retrieval Failures

| Failure | Impact | Recovery |
|---------|--------|----------|
| BM25 timeout (>5s) | Lose keyword matches | RRF proceeds with Dense + Graph only; quality degrades on exact-match queries |
| Dense retriever OOM | Lose semantic matches | Fallback to BM25 only; MRR drops ~X% (from benchmark) |
| All retrievers fail | No context for generation | Return error to user; do NOT generate without context (hallucination risk) |
| RRF tie (same score) | Arbitrary ordering | Break ties by document recency; prevents stale content promotion |

## Supervisor Failures

| Failure | Impact | Recovery |
|---------|--------|----------|
| Supervisor misroutes simple query to multi-agent | 2-3x latency, 2x cost | QualityAgent catches if answer is fine; overhead but correct |
| Supervisor loops (routes back to same agent) | Infinite loop risk | Max 3 supervisor decisions per query; exceeded → fallback to simple RAG |
| Agent raises exception | Supervisor hangs | Per-agent timeout (10s); on timeout, supervisor skips agent and proceeds |
| QualityAgent always rejects | Infinite retry | Max 2 quality retries; exceeded → route to human review |

## Cost Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| LLM cost spike (prompt injection → huge context) | Budget blown on single query | Per-query cost limit ($0.10); exceeded → reject before LLM call |
| Rate limit cascade (429 from all providers) | All queries fail | Circuit breaker opens → queue backpressure → client gets 503 with retry-after |
```

### Step 4: Restructure README to emphasize 2 deep patterns

In the agentic-rag README, restructure the "10 Agentic Design Patterns" section:

**Before** (current): flat table with all 10 equal.

**After**: Two highlighted patterns with depth, then "Also Implemented" for the rest.

```markdown
## Deep-Dive Patterns

### Parallel Retrieval with Reciprocal Rank Fusion

[Architecture diagram of BM25 + Dense + Graph → RRF merge]

**Why RRF over linear combination?** RRF is rank-based, not score-based. BM25 scores and dense cosine similarities are on different scales — normalizing them introduces a hyperparameter (α weight). RRF avoids this by using only rank positions.

**Parameter sensitivity:** k=60 achieves X% MRR, outperforming k=20 by X% and k=100 by X%. See `eval_results/rrf_sensitivity.json`.

**Failure modes:** If one retriever times out, RRF degrades gracefully — MRR drops X% with 2/3 retrievers vs X% with 1/3. See `docs/failure-modes.md`.

### Supervisor Multi-Agent Orchestration

[Architecture diagram of Supervisor → Research/Analysis/Quality agents]

**Why dynamic routing over hardcoded?** The supervisor reads AgentRegistry capabilities at decision time. Adding a new agent = 1 line of code. The prompt stays current without code changes.

**Routing accuracy:** X% on our 50-query benchmark. Misroutes cost X extra steps on average but are self-correcting via QualityAgent feedback.

**When NOT to use supervisor:** Simple factual queries. Our benchmark shows simple RAG is Xms faster and X% cheaper with equivalent quality on simple queries.

## Also Implemented

| # | Pattern | Module | One-line description |
|---|---------|--------|---------------------|
| 1 | Tool Use | `retriever.py` | Agents call external tools as structured functions |
| 2 | Reflection | `hallucination_checker.py` | Pipeline checks own outputs, decides retry/proceed |
| ... | ... | ... | ... |
```

### Step 5: Run all tests
```bash
cd /Users/joelle/Projects/side-projects/agentic-rag
source .venv/bin/activate
pytest tests/unit/ -v
```

### IMPORTANT NOTES
- Read ALL source files before writing any code. The retriever APIs, supervisor interface, and state schema must match exactly.
- Keep benchmark scripts under 150 lines each.
- The failure modes doc should be honest — include real weaknesses, not just "everything recovers."
- The README restructure should NOT remove any patterns — just de-emphasize 8 and highlight 2.
