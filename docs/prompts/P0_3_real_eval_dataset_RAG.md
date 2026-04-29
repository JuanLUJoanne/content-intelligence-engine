# P0-3: Add Real Eval Dataset + Retrieval Benchmarks (agentic-rag)

## Context
Project at `/Users/joelle/Projects/side-projects/agentic-rag`. It implements 10 agentic RAG patterns with LangGraph. Currently the eval section in README shows DummyLLM heuristic scores (~0.50 faithfulness, ~0.45 answer relevancy) with a disclaimer that "Real scores require live API evaluation." We need a real dataset and retrieval benchmarks to make the project credible.

Key files to read first:
- `src/retrieval/parallel_retriever.py` — BM25 + Dense + Graph retrieval with RRF merge
- `src/retrieval/bm25_retriever.py` — BM25 implementation
- `src/retrieval/dense_retriever.py` — Dense vector retrieval
- `src/eval/ragas_eval.py` — RAGAS-style evaluation
- `src/eval/comparative_eval.py` — Simple vs Multi-Agent comparison
- `src/graph/simple_workflow.py` — Corrective RAG workflow
- `src/graph/multi_agent_workflow.py` — Multi-agent supervisor workflow
- `src/graph/state.py` — State definition
- `src/utils/llm.py` — DummyLLM factory

## What to do

### Step 1: Create QA eval dataset
Create `eval_data/qa_100.jsonl` with 100 question-answer pairs. These should be standalone questions that a RAG system would answer. Each line:

```json
{
  "id": "qa-001",
  "question": "What is the difference between BM25 and dense retrieval?",
  "expected_answer": "BM25 is a sparse retrieval method based on term frequency and inverse document frequency, excelling at exact keyword matches. Dense retrieval uses neural embeddings to capture semantic similarity, handling paraphrases and synonyms better. BM25 is faster and more interpretable; dense retrieval captures meaning but requires vector infrastructure.",
  "difficulty": "simple",
  "category": "factual",
  "relevant_keywords": ["bm25", "dense", "retrieval", "embedding", "sparse"]
}
```

Requirements:
- 40 "simple" factual questions (single fact lookup)
- 30 "comparison" questions (compare two concepts)
- 20 "multi_hop" questions (require combining multiple facts)
- 10 "ambiguous" questions (could be interpreted multiple ways — tests routing)
- Categories: `factual`, `comparison`, `multi_hop`, `ambiguous`
- Topics: AI/ML, RAG systems, information retrieval, NLP, LLM patterns (domain-relevant)
- Include `relevant_keywords` for each — used to verify BM25 retrieval quality

### Step 2: Create retrieval benchmark script
Create `scripts/benchmark_retrieval.py`:

This script benchmarks 3 retrieval strategies against the eval dataset:
1. **BM25 only** — `src/retrieval/bm25_retriever.py`
2. **Dense only** — `src/retrieval/dense_retriever.py`
3. **RRF merged** — `src/retrieval/parallel_retriever.py` (BM25 + Dense + Graph, merged with RRF)

Read the actual retriever interfaces first. Each retriever should:
1. Index a set of synthetic documents (create 200 short docs covering the eval topics)
2. For each question in `eval_data/qa_100.jsonl`, retrieve top-5 docs
3. Measure:
   - **Retrieval latency** (ms per query)
   - **Keyword overlap** (what % of `relevant_keywords` appear in retrieved docs)
   - **Rank position** of the most relevant doc

Output to `eval_results/benchmark_retrieval.json`:
```json
{
  "benchmark": "retrieval_comparison",
  "dataset": "eval_data/qa_100.jsonl",
  "document_count": 200,
  "query_count": 100,
  "results": {
    "bm25_only": {
      "avg_latency_ms": X,
      "keyword_overlap_at_5": 0.XX,
      "mrr": 0.XX
    },
    "dense_only": {
      "avg_latency_ms": X,
      "keyword_overlap_at_5": 0.XX,
      "mrr": 0.XX
    },
    "rrf_merged": {
      "avg_latency_ms": X,
      "keyword_overlap_at_5": 0.XX,
      "mrr": 0.XX
    }
  },
  "rrf_parameters": {
    "k": 60,
    "note": "k=60 is standard RRF constant; higher k = more rank equality"
  },
  "conclusion": "RRF achieves X% better MRR than best single retriever at Y% latency overhead"
}
```

### Step 3: Create workflow comparison benchmark
Create `scripts/benchmark_workflows.py`:

Compare simple Corrective RAG vs Multi-Agent Supervisor on the 100 QA pairs:

For each question, run through both workflows (using DummyLLM) and measure:
- **Answer produced** (yes/no — did the workflow produce a final answer?)
- **Steps taken** (how many nodes executed?)
- **Retries triggered** (query rewrites, hallucination retries)
- **Cost estimate** (from cost tracker)
- **Latency** (wall clock time)
- **Quality score** (from RAGEvaluator if available)

Output to `eval_results/benchmark_workflows.json`:
```json
{
  "benchmark": "workflow_comparison",
  "query_count": 100,
  "results": {
    "simple_corrective_rag": {
      "completion_rate": 0.XX,
      "avg_steps": X.X,
      "avg_retries": X.X,
      "avg_cost": 0.XXXX,
      "avg_latency_ms": XX,
      "avg_quality": 0.XX
    },
    "multi_agent_supervisor": {
      "completion_rate": 0.XX,
      "avg_steps": X.X,
      "avg_retries": X.X,
      "avg_cost": 0.XXXX,
      "avg_latency_ms": XX,
      "avg_quality": 0.XX
    }
  },
  "by_difficulty": {
    "simple": {"winner": "simple_rag", "reason": "lower cost, same quality"},
    "comparison": {"winner": "multi_agent", "reason": "better decomposition"},
    "multi_hop": {"winner": "multi_agent", "reason": "parallel retrieval"},
    "ambiguous": {"winner": "multi_agent", "reason": "supervisor routing"}
  }
}
```

### Step 4: Update README
Replace the current "Eval Results (DummyLLM baseline)" section with:

```markdown
## Eval Results

Measured on 100 QA pairs (`eval_data/qa_100.jsonl`) across 4 difficulty levels.

### Retrieval Strategy Comparison

| Strategy | MRR@5 | Keyword Overlap | Latency |
|----------|-------|-----------------|---------|
| BM25 only | X.XX | X% | Xms |
| Dense only | X.XX | X% | Xms |
| RRF (BM25 + Dense + Graph) | X.XX | X% | Xms |

RRF with k=60 achieves X% better MRR than the best single retriever.

### Workflow Comparison (DummyLLM)

| Metric | Simple RAG | Multi-Agent | Delta |
|--------|-----------|-------------|-------|
| Completion rate | X% | X% | |
| Avg steps | X.X | X.X | +X.X overhead |
| Avg cost | $X.XXXX | $X.XXXX | +X% |
| Avg latency | Xms | Xms | +Xms |

**By difficulty:**
- Simple queries: Simple RAG wins (lower cost, same quality)
- Multi-hop queries: Multi-Agent wins (parallel retrieval + decomposition)
- Ambiguous queries: Multi-Agent wins (supervisor routing)

_All numbers from DummyLLM. Run `python scripts/benchmark_retrieval.py` and `python scripts/benchmark_workflows.py` to reproduce._
```

### Step 5: Run all tests
```bash
cd /Users/joelle/Projects/side-projects/agentic-rag
source .venv/bin/activate
pytest tests/unit/ -v
```

### IMPORTANT NOTES
- Read ALL source files before writing benchmark code. Match actual APIs.
- Use DummyLLM for everything — no API keys needed.
- The synthetic documents for retrieval benchmark should be short (2-3 sentences each), covering AI/ML/RAG topics.
- Keep scripts simple and readable. Under 200 lines each.
- Commit `eval_data/` and `eval_results/` — these are evidence, not artifacts.
