# Interview Preparation: After Portfolio Updates

## Does the Resume Need Updates?

**No.** The resume bullets from the previous session are already written to be defensible. What changes is your ABILITY to back them up:

| Resume Bullet | Before These Changes | After These Changes |
|---|---|---|
| "Built failure-type-aware routing" | Can explain code, no numbers | "98.5% compliance, measured on 50 records" |
| "Heuristic routing reduced cost 73%" | Claimed in README, no proof | "Benchmark shows heuristic = X% of premium quality at Y% cost" |
| "Error-feedback retries" | Can explain pattern | "+X pp fix rate vs blind retry, measured on 20 failure cases" |
| "LLM-as-Judge evaluation" | Can explain judge design | "X% agreement with human labels, calibrated on 10 records" |
| "RRF parallel retrieval" | Can explain RRF formula | "k=60 optimal, X% better MRR than best single retriever" |
| "Supervisor orchestration" | Can explain routing | "X% routing accuracy, misroutes cost X extra steps" |
| "AI observability (Langfuse)" | No code evidence | Real Langfuse wrapper + metrics module in codebase |
| "Containerized deployment" | No Dockerfile | Working Docker + docker-compose |

**The resume stays the same. The evidence behind it gets dramatically stronger.**

## What to Prepare Instead: Interview Depth Cheat Sheet

For each resume bullet, prepare a 3-layer answer:

### Layer 1: "What" (30 seconds)
The one-sentence description. You already have this.

### Layer 2: "Why + Tradeoff" (60 seconds)
Why this design over alternatives, with quantified tradeoffs.

### Layer 3: "Failure Mode + Scale" (60 seconds)
How it breaks, how you'd fix it at scale.

---

### Example: Failure-Type Routing

**Layer 1:** "We classify LLM validation failures into structural failures and field errors, routing them to different queues with different recovery strategies."

**Layer 2:** "Structural failure means the LLM returned plain text or completely wrong schema — no amount of retrying fixes this, it indicates a prompt regression, so we route to an engineering queue. Field error means valid JSON but wrong enum value — feeding the exact Pydantic error back to the LLM gives us 98.5% fix rate in 3 attempts, versus 91% with blind retries. The tradeoff is prompt length increases on retry, but the fix rate improvement more than compensates."

**Layer 3:** "At scale, the risk is engineering queue growth — if a prompt regression affects 100% of records, we get a flood. The mitigation is: alert on engineering queue growth rate > X/min, auto-pause the batch, and the on-call engineer investigates the prompt version diff. We store prompt version in the audit trail so we can identify which change caused the regression."

---

### Key Follow-Up Questions to Prepare

1. **"Why not use an LLM to classify failure type?"**
   → "Because the classification is deterministic: check if output has ANY expected fields. An LLM adds latency and cost for a binary decision that Pydantic already tells us. Over-engineering."

2. **"How do you prevent the error-feedback prompt from growing unbounded?"**
   → "We cap at 3 retries and truncate the error message to 200 chars. After 3 failures, the record goes to human review — at that point, automated fixing is unlikely."

3. **"What if the judge LLM hallucinates a high score?"**
   → "Our calibration benchmark shows X% agreement with human labels. We also use fallback score 0.5 when the judge returns invalid JSON, which is conservative. In production, we'd add a second judge (different model) and flag disagreements."

4. **"Why hash(user_id) % 2 for A/B, not a proper experiment framework?"**
   → "For this stage, deterministic assignment is sufficient — same user always sees same variant, no database lookup needed. The weakness is Python's hash() is process-local and not stable across restarts in 3.12+. Production would use SHA256 or a feature flag service like LaunchDarkly."

5. **"What happens when your circuit breaker opens?"**
   → "Queue backpressure activates — workers stop pulling. Currently process-local so in production we'd need Redis-backed state or Istio service mesh circuit breaking. I documented this in the production deployment architecture doc."

6. **"How do you handle multi-tenant cost isolation?"**
   → "Current portfolio is single-tenant. Production design: tenant_id from JWT → per-tenant CostTracker budget in DynamoDB with conditional decrement → per-tenant rate limits in Redis sorted sets. I documented the full multi-tenant architecture in docs/production-deployment.md."

7. **"Your RRF k=60 — why not tune it?"**
   → "I ran a sensitivity analysis: k=1 through k=200. k=60 is the standard because it balances rank-position weighting without over-indexing on top-1 results. k<20 makes the fusion too sensitive to individual retriever quality; k>100 approaches uniform weighting which defeats the purpose. Our benchmark shows k=60 gives X% MRR which is within 1% of optimal."
