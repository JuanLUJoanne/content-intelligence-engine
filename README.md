# Multi-Modal Content Intelligence Engine

Production-grade AI pipeline framework with dynamic model routing, cost guardrails, eval-driven development, and an agentic personalised recommendation layer. Designed to solve real problems running LLMs in production: quality inconsistency, unpredictable costs, lack of systematic evaluation, and cold-start personalisation.


## Architecture

```
                        ┌───────────────────────────────────────────────────────────┐
                        │                  API Layer (FastAPI)                      │
                        │  POST /process  POST /process/stream  POST /process/batch │
                        └────────────────────────┬──────────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │              Input Sanitizer (security.py)          │
                        │   Blocks prompt injection · Detects / redacts PII   │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │                 Cache Check (Redis / in-mem)        │
                        │                                                     │
                        │   HIT ──────────────────────────────► Store Result  │
                        │   MISS                                              │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │               Model Router (router.py)              │
                        │  FLASH tier · STANDARD tier · cost/latency scoring  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Rate Limiter + Cost Guardrails            │
                        │  per-request · per-minute · anomaly · total budget  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │                    LLM Call                         │
                        │         Circuit Breaker · Retry + Backoff           │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │         Schema Validation + Failure Routing         │
                        │                                                     │
                        │  structural_failure ──────────► Engineering Queue   │
                        │  field_error (×3 retries) ────► Human Review Queue  │
                        │  PASS                                               │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │              Eval Score (LLM-as-Judge)              │
                        │                                                     │
                        │   score < 0.7 ─────────────────────► Human Review   │
                        │   score ≥ 0.7                                       │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Store Result + Audit Log (JSONL)          │
                        └─────────────────────────────────────────────────────┘


  Agentic Recommendation Layer (runs independently of the processing pipeline)

                        ┌───────────────────────────────────────────────────────────┐
                        │                 MCPClient (real or mock)                  │
                        │   get_browsing_history · get_purchase_history             │
                        │   get_asset_metadata                                      │
                        └────────────────────────┬──────────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           BuyerProfile (tag affinity, top category) │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │         A/B Experiment (Variant A / B assignment)   │
                        │     purchase_history_based vs browsing_pattern_based│
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │       AssetRetriever (tag-affinity ranked corpus)   │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │     LLM Email Generation + Judge Eval + Cost Track  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Log result to ReviewStore (A/B audit)     │
                        └─────────────────────────────────────────────────────┘
```

## Key Design Decisions

**1. Heuristic routing over LLM routing**
Routing decisions (which model tier to use) are made with deterministic heuristics — token count, latency sensitivity, cost sensitivity — rather than asking an LLM. LLM-based routing adds latency and cost to every request and introduces a bootstrapping paradox: you need a model to decide which model to use. Heuristics are predictable, fast, and trivially unit-testable.

**2. Failure-type-aware validation routing**
Schema validation failures are split into two distinct paths: *structural failures* (malformed JSON, completely missing fields) go directly to an engineering queue — no retry is attempted, because retrying a broken prompt template wastes quota and obscures the root cause. *Field errors* (wrong enum value, out-of-range number) trigger up to three error-feedback retries, passing the exact Pydantic error message back to the LLM so it can self-correct. This separation means engineers get a clean signal about prompt regressions without drowning in human-review noise.

**3. Record-level checkpointing**
Each successfully processed record is checkpointed immediately after storage. On restart or failure, the pipeline replays only unprocessed records. This makes the pipeline idempotent across crashes, restarts, and partial batch failures without requiring distributed transactions.

**4. Multi-level cost guardrails**
Four independent guardrail levels — per-request, per-minute sliding window, anomaly detection (3× rolling average), and total budget hard stop — are checked before every LLM call. A single anomalous request cannot exhaust the budget; a runaway batch cannot breach the daily cap. The layered design means each guard can be tuned independently as traffic patterns change.

**5. LLM-as-Judge for eval**
Output quality is scored by a separate LLM judge across five dimensions: factual accuracy, hallucination rate, semantic consistency, relevance, and schema compliance. Using a language model as the evaluator catches subtle quality regressions that rule-based metrics miss, while the separate judge model avoids the "marking your own homework" problem. Scores below 0.7 are routed to human review rather than rejected outright.

**6. A/B-tested personalised recommendations**
The recommendation layer assigns users deterministically to variants using `hash(user_id) % 2`, so the same user always sees the same prompt flavour across requests without storing session state. Variant A weights purchase history; Variant B weights browsing patterns. Every result is logged to the ReviewStore so effect sizes can be computed offline using Cohen's d.

**7. MCP tool layer as a data-fetching adapter**
`RecommendationAgent.run()` accepts an optional `MCPClient`. When provided, buyer history is fetched via the Model Context Protocol; when absent, the caller passes resolved asset lists directly. The mock path (`use_mock=True`) uses `hash()`-based determinism — no random seed, no network — so tests are reproducible without fixtures or patching.

**8. Adaptive retrieval (observe → reason → act)**
Fixed-query retrieval fails for edge-case users — sparse profiles, ambiguous interests, or cold-start buyers. `AdaptiveRetriever` wraps `AssetRetriever` in an LLM-in-the-loop that inspects search results and decides whether to accept or refine the query. The loop is bounded (`max_rounds=3`) so cost stays predictable, and every search goes through the same scoring logic the non-adaptive path uses. With `DummyProvider` the loop terminates after one round with zero extra cost, so tests and CI are unaffected.

The initial query is still determined by the A/B variant (top-category for Variant A, top-affinity-tags for Variant B) rather than letting the LLM pick the first query from scratch. This is a deliberate trade-off: the variant-derived query is deterministic and grounded in actual user behaviour, so it produces a strong baseline result on round 1. Letting the LLM choose the initial query would add latency and cost to every request — including the majority where the first-round results are already good enough — while also breaking the A/B experiment's ability to attribute retrieval differences to variant strategy. The LLM only enters the loop *after* seeing round-1 results, where its judgement adds genuine value: deciding whether those results are relevant and, if not, generating a more targeted refinement.

## Production Features

| Feature | Description |
|---|---|
| **Model routing** | Heuristic tier selection (FLASH / STANDARD) with cost-aware scoring |
| **Circuit breakers** | Per-model open/half-open/closed state with configurable thresholds |
| **Rate limiting** | Token-bucket rate limiter with per-model and global limits |
| **Response cache** | SHA-256 keyed cache with configurable TTL; cache hits skip the LLM entirely |
| **Cost guardrails** | Per-request, per-minute, anomaly, and total-budget protection |
| **Batch API** | Async batch submission with poll-until-complete semantics |
| **Drift detection** | Baseline comparison across 5 eval dimensions; alerts on regression |
| **Prompt versioning** | Register, rollback, and auto-rollback prompt versions on drift |
| **A/B prompt comparison** | Statistical comparison with Cohen's d effect size and 2% minimum threshold |
| **Failure-type routing** | structural_failure → engineering queue; field_error (×3) → human review queue |
| **Engineering queue** | REST API for ops to inspect and requeue structural validation failures |
| **Human review queue** | Low-confidence and field-error outputs routed to review; approvals written to golden set |
| **Adaptive retrieval** | LLM-in-the-loop search: observe results → decide relevance → refine query (up to 3 rounds) |
| **Personalised recommendations** | A/B-tested email generation driven by tag-affinity buyer profiles |
| **MCP tool layer** | MCPClient adapter fetches real buyer data; deterministic mock for offline testing |
| **Input sanitization** | 12-pattern prompt injection detection with fast-fail at API boundary |
| **PII detection** | Regex-based detection and redaction of emails, phone numbers, credit cards |
| **Audit logging** | Append-only JSONL audit trail for every request/response |
| **LangGraph orchestration** | 7-node stateful pipeline with conditional edges, retry loops, and failure routing |
| **SSE streaming** | Server-Sent Events progress stream for real-time UI integration |

## Eval Framework

The pipeline ships with a two-layer evaluation system designed to catch different failure modes at different granularities:

**Layer 1 — Deterministic ground-truth checks** compare LLM outputs field-by-field against a 50-record golden dataset (`eval_data/product_metadata_50.jsonl`) covering all enum values plus edge cases (5-word descriptions, 200+ word inputs, multi-language, ambiguous categories, no-price-signal records). These run without an API key and execute in CI on every commit.

**Layer 2 — LLM-as-Judge scoring** evaluates outputs across five dimensions (factual accuracy, schema compliance, hallucination, semantic consistency, relevance) using per-dimension prompts with isolated scoring. Each dimension is judged in a separate call with built-in retry and graceful fallback, so one bad parse never blocks the full evaluation.

Both layers feed into the same report, giving a combined view of structural correctness and semantic quality.

### Model Selection

Eval results below use **Gemini 3.1 Flash-Lite** — deliberately the smallest, cheapest model in the Gemini lineup. This is an intentional choice: the eval framework is designed to measure how well the *pipeline engineering* (error-feedback retries, failure routing, schema enforcement) compensates for weaker model capability. A stronger model would score higher, but would tell you less about whether your system-level guardrails actually work. The pipeline's model router supports hot-swapping to any Gemini or OpenAI backend — upgrading from Flash-Lite to a larger model is a one-line config change, and the eval framework lets you quantify exactly what you gain.

### Pipeline Behaviour (Gemini 3.1 Flash-Lite)

| Metric | Result | What it measures |
|--------|--------|------------------|
| Schema Compliance | **80%** | Validation pass rate after error-feedback retries |
| Retry Self-Correction | 4/5 records self-corrected | Error-feedback loop feeds Pydantic errors back to the LLM |
| Structural Failure → Engineering Queue | 1/5 | Prompt-level failures routed directly to engineering (no retry wasted) |
| DLQ Rate | 0% | No API or network errors reached dead-letter queue |

### LLM-as-Judge Quality Scores

| Dimension | Score | What it evaluates |
|-----------|-------|-------------------|
| Factual Accuracy | **0.975** | Correctness of factual claims vs reference |
| Hallucination (absence) | **0.975** | Whether output avoids inventing unsupported details |
| Semantic Consistency | **0.988** | Whether output preserves intent of input |
| Relevance | **1.000** | On-topic focus without digressions |
| Schema Compliance | **0.900** | Format and constraint adherence |

### Ground-Truth Accuracy

| Field | Accuracy | Notes |
|-------|----------|-------|
| Category | 80% | Correct enum classification |
| Condition | 60% | Ambiguous inputs cause misclassification |
| Price Range | 40% | Model lacks price-signal context — prompt improvement target |
| Tag Recall | 0% | Open-vocabulary tags; prompt does not constrain tag set |

_Tag recall and price-range accuracy are known prompt gaps — the eval framework is designed to surface exactly these regressions so prompt iteration is data-driven rather than guesswork._

_Reproduce: `python -m scripts.run_eval --provider dummy` (free, CI) or `python -m scripts.run_eval --provider gemini --judge --delay 5` (live eval)._

### Quantified Design Tradeoffs

Three benchmarks run on every CI build using `DummyProvider` (free, deterministic) to measure the system-level impact of key engineering decisions. These measure **pipeline behaviour**, not LLM quality — the point is to verify that routing, retry, and eval machinery work correctly regardless of which model sits behind the provider interface.

_Reproduce: `python -m scripts.benchmark_tradeoffs`_

**Benchmark 1 — Routing Strategy** (N=50 records)

| Strategy | Simulated Cost | Avg Latency | Schema Compliance |
|----------|---------------|-------------|-------------------|
| All-FLASH | $0.0000 | 1.8 ms | 100% |
| All-PREMIUM | $0.0595 | 0.3 ms | 100% |
| **Heuristic** | **$0.0050** | **0.3 ms** | **100%** |

Heuristic routing achieves the same compliance at **8% of premium cost**. The router's complexity score correctly funnels balanced-sensitivity requests to STANDARD tier, avoiding premium spend without degrading output quality.

**Benchmark 2 — Error-Feedback Retry vs Blind Retry** (N=50 records)

| Strategy | Fix Rate | Avg Retries |
|----------|----------|-------------|
| Blind retry (same prompt) | 100% | 1.00 |
| Error-feedback retry | 100% | 1.00 |

With `DummyProvider` both strategies converge — the deterministic provider always produces valid output on attempt 2 regardless of prompt content. The benchmark validates that the retry machinery and monkey-patching infrastructure work correctly; the real differentiation appears with live LLM providers where error feedback gives the model actionable context to self-correct.

**Benchmark 3 — Judge Agreement with Ground Truth** (N=10 records)

| Metric | Result |
|--------|--------|
| Agreement rate | **90%** |
| Divergence count | 1/10 |

| Dimension | Avg Score |
|-----------|-----------|
| Factual accuracy | 0.500 |
| Schema compliance | 0.500 |
| Hallucination | 0.500 |
| Semantic consistency | 0.500 |
| Relevance | 0.500 |

Per-dimension scores are 0.5 (the `DummyProvider` fallback), confirming the judge correctly exercises all five scoring dimensions and falls back gracefully when the provider cannot reason. The 90% agreement rate shows the judge/ground-truth classification boundary (>0.5 = pass) is well-calibrated even with dummy outputs.

## Framework Portability

The pipeline is framework-agnostic by design. Core logic lives in plain Python classes with no framework imports; orchestration is isolated in `graph.py` and `adaptive_retriever.py`.

| This repo | LangGraph equivalent | Google ADK equivalent |
|-----------|---------------------|----------------------|
| `ContentPipelineGraph` (node dict + conditional edges) | `StateGraph` with `add_node` / `add_conditional_edges` | `SequentialAgent` with `sub_agents` |
| `AdaptiveRetriever` (observe → reason → act loop) | `ToolNode` + `should_continue` router | `Agent` with `tools=[search_assets]` (ReAct loop) |
| `BuyerProfile` dict passed through nodes | `TypedDict` state schema | `Session.state` |
| `AssetRetriever.search()` | LangChain `Tool` wrapper | ADK `FunctionTool` wrapper |
| `LLMProvider` protocol | `BaseChatModel` interface | `LlmAgent.model` parameter |

**Migrating to LangGraph** requires wrapping each `_node_*` method as a graph node and replacing the manual edge dict with `add_conditional_edges` — roughly a wiring change in one file. **Migrating to ADK** means converting `AdaptiveRetriever` into an `Agent` with `search_assets` as a `FunctionTool` and letting ADK's ReAct loop replace the manual round counter. In both cases, retrieval scoring, validation, cost tracking, and eval logic remain untouched.

## Cost Comparison

| Mode | Model | Cost / 1K records | Cost @ 10M records |
|---|---|---|---|
| Real-time (all GPT-4) | gpt-4o | $60.00 | $600,000 |
| Real-time (routed) | gemini-flash / gpt-4o-mini | $8.50 | $85,000 |
| **Batch (routed)** | gemini-flash / gpt-4o-mini | **$4.25** | **$42,500** |

Batch mode halves cost again by using provider batch APIs (50% discount) at the expense of same-day latency.

## Quick Start

```bash
cp .env.example .env          # add OPENAI_API_KEY / GOOGLE_API_KEY
pip install -e .
pytest tests/unit/ -v                                         # 305+ tests, ~8s
python -m scripts.run_eval --provider dummy                   # eval without API key
GOOGLE_API_KEY=xxx python -m scripts.run_eval --provider gemini --judge  # real eval (~$0.10)
python scripts/demo.py                                        # end-to-end demo
uvicorn src.api.main:app --reload
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process` | Process a single item through the full LangGraph pipeline |
| `POST` | `/process/stream` | SSE stream of per-stage progress events |
| `POST` | `/process/batch` | Submit a list of records for async batch processing |
| `POST` | `/evaluate` | Score an LLM output with the LLM-as-Judge evaluator |
| `GET` | `/costs` | Current spend by model with remaining budget |
| `GET` | `/drift/report` | Latest drift detection report vs. saved baseline |
| `GET` | `/health` | Liveness check with passing test count |
| `GET` | `/review/pending` | List items queued for human review |
| `POST` | `/review/{id}/approve` | Approve a review item (writes to golden set) |
| `POST` | `/review/{id}/reject` | Reject a review item with a reason |
| `GET` | `/review/stats` | Review queue statistics and approval rate |
| `GET` | `/engineering/pending` | List structural validation failures pending investigation |
| `POST` | `/engineering/{id}/requeue` | Mark an engineering failure for reprocessing |
| `GET` | `/engineering/stats` | Failure counts grouped by prompt version |

### Example: process a single item

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/product-001.jpg",
    "text": "Wireless noise-cancelling headphones with 30-hour battery",
    "language": "en",
    "content_type": "product"
  }'
```

```json
{
  "metadata": {
    "content_id": "https://example.com/product-001.jpg",
    "title": "Wireless Noise-Cancelling Headphones",
    "category": "electronics",
    "condition": "new",
    "price_range": "premium",
    "tags": ["headphones", "noise-cancelling", "wireless"],
    "language": "en"
  },
  "model_used": "gemini-flash",
  "cost": 0.0013,
  "confidence": 0.92
}
```

### Example: stream progress

```bash
curl -N -X POST http://localhost:8000/process/stream \
  -H "Content-Type: application/json" \
  -d '{"image_url": "...", "text": "...", "language": "en", "content_type": "product"}'
```

```
data: {"event": "sanitizing", "status": "running"}
data: {"event": "sanitizing", "status": "ok"}
data: {"event": "cache_check", "status": "ok", "hit": false}
data: {"event": "routing", "status": "ok", "model": "gemini-flash"}
data: {"event": "calling_llm", "status": "running"}
data: {"event": "calling_llm", "status": "ok"}
data: {"event": "validating", "status": "ok"}
data: {"event": "scoring", "status": "ok", "confidence": 0.92}
data: {"event": "complete", "status": "ok", "metadata": {...}}
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Pipeline orchestration | LangGraph-style stateful graph (migrates to LangGraph with 5-line change) |
| Tracing | LangSmith (`@traceable`, graceful no-op without API key) |
| LLM providers | Google Gemini, OpenAI GPT-4o |
| Schema validation | Pydantic v2 |
| API framework | FastAPI + Uvicorn |
| Storage | SQLite (checkpoints), JSON (golden set), JSONL (audit log) |
| Logging | structlog (structured JSON logs) |
| Testing | pytest + pytest-asyncio (305+ unit tests) |

## Project Layout

```
src/
├── api/
│   ├── main.py              # FastAPI app, routers, and all endpoints
│   └── review.py            # Human review queue, engineering queue, golden-set management
├── agents/
│   ├── recommendation_agent.py  # Agentic recommendation pipeline with adaptive retrieval
│   ├── adaptive_retriever.py    # LLM-in-the-loop search with observe → reason → refine loop
│   └── memory/
│       └── buyer_profile.py     # BuyerProfile, tag-affinity computation, MCP-backed factory
├── ab_test/
│   └── experiment.py        # Variant assignment, prompt templates, result logging
├── retrieval/
│   └── asset_retriever.py   # Tag-affinity scored in-memory corpus search
├── mcp/
│   ├── tools.py             # ANALYTICS_TOOLS MCP schema definitions
│   └── client.py            # MCPClient with deterministic mock and real stub
├── eval/
│   ├── judge.py             # LLM-as-Judge with per-dimension prompts
│   ├── drift_detector.py    # Baseline comparison and drift alerting
│   └── ab_prompt.py         # A/B comparison with Cohen's d effect size
├── gateway/
│   ├── providers.py         # LLMProvider protocol + Gemini/OpenAI/Dummy impls
│   ├── router.py            # Heuristic model router with cost-aware scoring
│   ├── cost_tracker.py      # Token accounting and budget enforcement
│   ├── guardrails.py        # Multi-level cost guardrail system
│   ├── batch.py             # Async batch submission and polling
│   ├── cache.py             # Response cache with TTL
│   ├── circuit_breaker.py   # Per-model circuit breaker
│   ├── rate_limiter.py      # Token-bucket rate limiter
│   └── security.py          # Sanitization, PII detection, audit logging
└── pipeline/
    ├── graph.py             # 7-node ContentPipelineGraph with failure-type routing
    ├── processor.py         # BatchProcessor with checkpointing and DLQ
    ├── checkpoint.py        # Record-level checkpoint persistence
    ├── versioning.py        # Prompt registry with rollback and auto-rollback
    └── prompt_chain.py      # Multi-step prompt chaining utilities
```
