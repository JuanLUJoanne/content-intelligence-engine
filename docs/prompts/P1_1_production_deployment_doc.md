# P1-1: Add Production Deployment Architecture Doc (content-intelligence-engine)

## Context
This project uses in-memory queues, SQLite, JSONL — all dev-grade. Tier 1 interviewers will ask "how does this scale to 10M QPS?" The code doesn't need to change (it's a portfolio project), but we need a doc that shows we KNOW the production gaps and have a concrete upgrade plan.

## What to do

Create `docs/production-deployment.md` with the following content. Write it as a technical design doc, not marketing material. Be specific with technology choices and justify each one.

### Structure:

```markdown
# Production Deployment Architecture

This document describes how the current single-process portfolio implementation maps to a production deployment handling 10M+ records/day with multi-tenant isolation.

## Component Migration Matrix

| Component | Portfolio (Current) | Production | Migration Rationale |
|-----------|-------------------|------------|---------------------|
| Message Queue | `asyncio.Queue` (in-memory) | **Amazon SQS / Google Pub/Sub** | At-least-once delivery, DLQ with redrive policy, cross-service decoupling. SQS chosen over Kafka because our workload is task-queue (fan-out to workers), not event-stream (ordered log replay). |
| Response Cache | SQLite with lazy TTL | **Redis Cluster** (ElastiCache) | Multi-instance read sharing, sub-ms latency, active TTL expiry. Redis over Memcached because we need TTL per key and atomic operations for rate limiting. |
| Audit Trail | JSONL append-only file | **CloudWatch Logs → S3 → Athena** | Queryable, alertable, compliant retention. JSONL format preserved (newline-delimited JSON → CloudWatch JSON log events are 1:1 compatible). |
| Tracing | structlog + @traceable decorator | **Langfuse** (LLM-specific) + **Datadog APM** (infra) | Langfuse tracks generation quality, cost, latency per LLM call. Datadog tracks request-level distributed traces across services. Two layers because LLM observability needs (prompt versioning, eval scores, cost attribution) differ from infrastructure observability (p99 latency, error rates, CPU). |
| Circuit Breaker | Process-local dict | **Service mesh** (Istio) or **Redis-backed** shared state | Multi-instance deployment needs shared failure counts. Istio preferred if already on K8s; Redis-backed if running on ECS/Lambda. |
| Checkpoint | JSON file (temp + rename) | **DynamoDB** (conditional writes) | Exactly-once processing via conditional `PutItem` with `attribute_not_exists`. File rename is atomic on single machine but not across instances. |
| Cost Guardrails | In-process Decimal tracking | **Redis sorted sets** + **DynamoDB** for budget ledger | Per-minute sliding window in Redis (ZRANGEBYSCORE), total budget in DynamoDB with conditional decrement. |
| Batch Processing | AsyncQueueProcessor with asyncio.Queue | **AWS Step Functions** or **Temporal** | Long-running batch (10K+ records) needs: resume from failure, visibility into progress, timeout per step, parallel execution with controlled concurrency. |

## Scaling Bottlenecks & Solutions

### Bottleneck 1: LLM Provider Rate Limits
At 10M records/day ≈ 115 QPS sustained. Gemini Flash allows ~1000 QPS, so single-provider is fine. But:
- **Burst handling**: Queue absorbs spikes; workers pull at steady rate
- **Multi-provider failover**: Route to OpenAI when Gemini circuit breaker opens
- **Batch API**: Gemini/OpenAI batch endpoints give 50% cost reduction for non-latency-sensitive workloads

### Bottleneck 2: Schema Validation Retries
At 7.5% first-attempt failure rate, 115 QPS × 0.075 = ~9 retries/sec. Each retry is another LLM call.
- **Solution**: Error-feedback retries keep this to max 3 attempts. After 3, route to human review (not more retries).
- **Monitoring**: Alert on retry_rate > 15% (indicates prompt regression, not data issue)

### Bottleneck 3: Human Review Queue Growth
If 5% of records need review at 10M/day = 500K pending reviews/day. Unsustainable.
- **Solution 1**: Auto-approve records where eval_score > 0.6 AND retry_count < 2 (confidence threshold)
- **Solution 2**: Batch review UI (approve/reject 50 at once with sampling)
- **Solution 3**: Use approved records to fine-tune → reduce review rate over time

## Multi-Tenant Isolation

```
Request → API Gateway (tenant_id from JWT)
  → SQS Queue (per-tenant or shared with tenant_id attribute)
    → Worker (reads tenant_id, applies tenant-specific config)
      → LLM Call (tenant-specific budget, model preference, prompt version)
        → Results (tagged with tenant_id, stored in tenant-partitioned table)
```

- **Cost isolation**: Each tenant has its own CostTracker budget in DynamoDB
- **Rate isolation**: Per-tenant rate limits in Redis (ZRANGEBYSCORE with tenant prefix)
- **Data isolation**: All cache keys, audit logs, and review items prefixed with tenant_id
- **Model isolation**: Tenant config specifies allowed models and quality thresholds

## SLO / SLA Definitions

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Schema compliance | ≥ 98% | % records passing validation (after retries) | < 95% triggers page |
| P95 latency (single record) | ≤ 2s | End-to-end from API receipt to response | > 3s triggers warning |
| P99 latency (single record) | ≤ 5s | Includes retry time | > 8s triggers page |
| Cost per record | ≤ $0.002 | Monthly rolling average | > $0.003 triggers review |
| Human review rate | ≤ 5% | % records routed to review queue | > 10% triggers prompt investigation |
| Availability | 99.9% | API health check success rate | < 99.5% triggers page |

## Autoscaling Behavior

**Worker autoscaling** (ECS/K8s HPA):
- Scale-up trigger: SQS `ApproximateNumberOfMessagesVisible` > 1000 for 2 minutes
- Scale-down trigger: Queue empty for 5 minutes
- Min instances: 2 (availability), Max: 20 (cost cap)
- Scale-up cooldown: 60s, Scale-down cooldown: 300s (avoid thrashing)

**API autoscaling**:
- Scale on CPU > 70% sustained 2 minutes
- Min: 2 instances behind ALB
- Health check: `/health` endpoint every 10s

## Disaster Recovery

- **Queue**: SQS messages retained 14 days. DLQ has separate alarm + manual reprocessing script.
- **Cache**: Redis loss is non-fatal (cache miss → LLM call, slightly higher cost). No replication needed for cache-only use.
- **Checkpoints**: DynamoDB with point-in-time recovery. Batch can resume from last checkpoint.
- **Audit trail**: S3 with versioning + cross-region replication for compliance.
```

### IMPORTANT NOTES
- This is a DOCUMENT, not code. Write it as a technical design doc.
- Be opinionated — pick specific AWS services and justify why.
- Include cost estimates where possible (e.g., "SQS at 10M messages/day ≈ $4/day")
- Keep it under 200 lines. Dense, specific, no fluff.
- The point is to show the author understands production constraints WITHOUT over-engineering the portfolio code.
