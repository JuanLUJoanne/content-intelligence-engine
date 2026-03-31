# Production Deployment Architecture

This document describes how the current single-process portfolio implementation maps to a production deployment handling 10M+ records/day with multi-tenant isolation.

## Component Migration Matrix

| Component | Portfolio (Current) | Production | Migration Rationale |
|-----------|-------------------|------------|---------------------|
| Message Queue | `asyncio.Queue` (in-memory) | **Amazon SQS** | At-least-once delivery, DLQ with redrive policy, cross-service decoupling. SQS over Kafka because our workload is task-queue (fan-out to workers), not event-stream (ordered log replay). SQS at 10M messages/day ≈ $4/day. |
| Response Cache | SQLite with lazy TTL | **Redis Cluster** (ElastiCache) | Multi-instance read sharing, sub-ms latency, active TTL expiry. Redis over Memcached because we need TTL per key and atomic operations for rate limiting. r6g.large cluster ≈ $150/month. |
| Audit Trail | JSONL append-only file | **CloudWatch Logs → S3 → Athena** | Queryable, alertable, compliant retention. JSONL format preserved — newline-delimited JSON maps 1:1 to CloudWatch JSON log events. S3 storage at 10M records/day (~2 GB/day) ≈ $2/month; Athena queries billed per scan. |
| Tracing | structlog + `@traceable` | **Langfuse** (LLM-specific) + **Datadog APM** (infra) | Langfuse tracks generation quality, cost, latency per LLM call with prompt versioning and eval score history. Datadog tracks request-level distributed traces. Two layers because LLM observability needs (prompt versioning, eval scores, cost attribution) differ fundamentally from infrastructure observability (p99 latency, error rates, CPU). |
| Circuit Breaker | Process-local dict | **Redis-backed** shared state | Multi-instance deployment needs shared failure counts. Redis `INCR` with TTL gives a distributed counter that decays automatically. Istio service mesh is an alternative if already on K8s, but adds operational complexity we don't need at this scale. |
| Checkpoint | JSON file (temp + rename) | **DynamoDB** (conditional writes) | Exactly-once processing via conditional `PutItem` with `attribute_not_exists(record_id)`. File rename is atomic on a single machine but not across instances. On-demand DynamoDB at 10M writes/day ≈ $13/day. |
| Cost Guardrails | In-process Decimal tracking | **Redis sorted sets** + **DynamoDB** budget ledger | Per-minute sliding window in Redis (`ZRANGEBYSCORE` on timestamped entries), total budget in DynamoDB with conditional decrement to prevent overspend under concurrent writes. |
| Batch Processing | `AsyncQueueProcessor` with `asyncio.Queue` | **AWS Step Functions** | Long-running batches (10K+ records) need resume-from-failure, per-step timeout, progress visibility, and controlled concurrency. Step Functions over Temporal because we don't need custom retry policies complex enough to justify self-hosting a Temporal cluster. Step Functions at 10M state transitions/day ≈ $25/day. |

## Scaling Bottlenecks & Solutions

### Bottleneck 1: LLM Provider Rate Limits

10M records/day ≈ 115 QPS sustained. Gemini Flash allows ~1000 QPS, so single-provider handles steady state. Burst and failure scenarios require more:

- **Burst absorption**: SQS decouples ingestion from processing. Workers pull at a steady rate regardless of ingest spikes.
- **Multi-provider failover**: Router already supports Gemini and OpenAI. When Gemini circuit breaker opens, traffic shifts to OpenAI automatically. Adding a provider is one class implementing the `LLMProvider` protocol.
- **Batch API for backfills**: Gemini/OpenAI batch endpoints give 50% cost reduction. The existing `BatchProcessor` submits and polls — production version submits to provider batch API instead of per-record calls.

### Bottleneck 2: Schema Validation Retries

At 7.5% first-attempt failure rate (observed in eval), 115 QPS × 0.075 ≈ 9 retries/sec. Each retry is another LLM call.

- **Bounded retries**: Error-feedback retries cap at 3 attempts. After 3, the record routes to human review — not more retries. This bounds retry amplification to 1.225× worst case.
- **Prompt regression detection**: Alert when `retry_rate > 15%` over a 5-minute window. A spike in retries signals a prompt regression (systematic), not a data issue (random). The existing drift detector surfaces this.

### Bottleneck 3: Human Review Queue Growth

5% review rate at 10M/day = 500K pending reviews/day. No team can review that manually.

- **Auto-approve gate**: Records with `eval_score > 0.6` AND `retry_count < 2` are auto-approved with a `confidence: auto` flag. This covers the majority of borderline cases where the judge scored low due to conservative thresholds, not actual quality issues.
- **Batch review UI**: Approve/reject 50 records at once with statistical sampling — reviewer checks 5 of 50, applies decision to the batch if all 5 pass.
- **Feedback loop**: Approved records feed back into the golden set. Over time, the eval baseline tightens and the auto-approve gate handles more cases, reducing human load.

## Multi-Tenant Isolation

```
Request → API Gateway (tenant_id from JWT)
  → SQS Queue (shared, with tenant_id message attribute)
    → Worker (reads tenant_id, loads tenant config)
      → LLM Call (tenant-specific budget, model preference, prompt version)
        → Results (tenant-partitioned storage, tenant-scoped audit log)
```

- **Cost isolation**: Each tenant has its own budget row in DynamoDB. The `CostTracker` already accepts a budget parameter — production version reads it from tenant config.
- **Rate isolation**: Per-tenant rate limits in Redis using `ZRANGEBYSCORE` with `tenant:{id}:` key prefix. One tenant's burst cannot starve another.
- **Data isolation**: All cache keys, audit log entries, checkpoint records, and review items include `tenant_id` as partition key. Athena queries filter by tenant partition in S3.
- **Model isolation**: Tenant config specifies allowed model tiers and quality thresholds. A cost-sensitive tenant can restrict to FLASH-only; a quality-sensitive tenant can force STANDARD minimum.

## SLO / SLA Definitions

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Schema compliance | ≥ 98% | % records passing validation after retries | < 95% triggers page |
| P95 latency (single) | ≤ 2s | End-to-end API receipt → response | > 3s triggers warning |
| P99 latency (single) | ≤ 5s | Includes retry time | > 8s triggers page |
| Cost per record | ≤ $0.002 | Monthly rolling average | > $0.003 triggers review |
| Human review rate | ≤ 5% | % records routed to review queue | > 10% triggers prompt investigation |
| Availability | 99.9% | `/health` success rate (5-minute windows) | < 99.5% triggers page |

## Autoscaling Behaviour

**Worker autoscaling** (ECS Service Auto Scaling):

- Scale-up: SQS `ApproximateNumberOfMessagesVisible` > 1000 for 2 minutes
- Scale-down: Queue depth = 0 for 5 minutes
- Min instances: 2 (availability), Max: 20 (cost cap)
- Cooldown: 60s up, 300s down (asymmetric to avoid thrashing)

**API autoscaling** (ECS behind ALB):

- Scale on CPU > 70% sustained 2 minutes
- Min: 2 instances, Max: 10
- Health check: `/health` every 10s, 3 consecutive failures → drain

## Disaster Recovery

- **Queue**: SQS retains messages for 14 days. DLQ has a separate CloudWatch alarm + a reprocessing Lambda that moves messages back to the main queue after investigation.
- **Cache**: Redis loss is non-fatal — cache miss triggers an LLM call, temporarily increasing cost. No replication needed for cache-only data. Estimated cost impact of full cache loss: ~2× spend for 1 TTL period (~1 hour).
- **Checkpoints**: DynamoDB with point-in-time recovery enabled. Batch processing resumes from the last checkpointed record with zero reprocessing of completed items.
- **Audit trail**: S3 with versioning and cross-region replication to a compliance account. Lifecycle policy: Standard → IA at 30 days → Glacier at 90 days. Total retention: 7 years (configurable per tenant for regulatory requirements).

## Cost Estimate Summary (10M records/day)

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| SQS | ~$120 | 300M messages/month (including retries and DLQ) |
| ElastiCache (Redis) | ~$150 | r6g.large, single-AZ for cache workload |
| DynamoDB | ~$400 | On-demand, checkpoints + budget ledger |
| Step Functions | ~$750 | 300M state transitions/month |
| ECS (workers) | ~$500 | 2–20 instances, avg 5 × c6g.medium |
| ECS (API) | ~$200 | 2–10 instances, avg 3 × c6g.medium |
| LLM API (Gemini Flash, routed) | ~$25,500 | $0.085/1K records × 300M/month |
| **Total infrastructure** | **~$2,120/month** | Excluding LLM API costs |
| **Total with LLM** | **~$27,620/month** | LLM cost dominates by 12× |

LLM API cost is 92% of total spend. Infrastructure optimization has diminishing returns — the highest-leverage cost reduction is model routing (already implemented) and batch API usage (50% discount on non-latency-sensitive workloads).
