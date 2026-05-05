"""Benchmark integration test: node deduplication latency at scale with LadybugDB.

This test measures ``add_episode`` wall-clock time when the graph already contains
50 000+ entities in a single group, which is the scale at which the 70K→80K timeout
cliff was observed in production.

Running the benchmark
---------------------
The test is **skipped by default** unless ``GRAPHITI_BENCH_LADYBUG_DB`` is set.
Set it to a path to a pre-seeded LadybugDB ``db/`` directory (the directory
that contains ``graphiti.kz``)::

    GRAPHITI_BENCH_LADYBUG_DB=/path/to/seeded/db \\
    OPENAI_API_KEY=sk-... \\
    pytest tests/driver/test_dedup_benchmark_int.py -v -s --timeout=600 -m benchmark

The test will also skip if the configured database contains fewer than
``BENCH_MIN_ENTITY_COUNT`` (50 000) entities — which avoids false passes on an
under-populated fixture.

Seeding the database
--------------------
If you do not have a pre-seeded database, you can create one by running the
main graphiti ingest pipeline against a large corpus and pointing
``GRAPHITI_BENCH_LADYBUG_DB`` at the resulting ``db/`` directory.

SLO
---
p95 ``add_episode`` wall time ≤ 60 seconds at 50K+ entities in a single group
(see :data:`BENCH_P95_TARGET_SECONDS`).

With *full* LLM dedup (default config), this SLO may not be achievable at
80K+ entities due to LadybugDB HNSW write overhead combined with large LLM
prompt size. Use ``DeduplicationConfig(skip_llm_dedup=True)`` or
``DeduplicationConfig(cosine_min_score=0.85)`` to reduce latency.
"""

import logging
import os
import statistics
import time
from datetime import datetime, timezone

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

BENCH_MIN_ENTITY_COUNT = 50_000
BENCH_P95_TARGET_SECONDS = 60.0
BENCH_GROUP_ID = 'bench-group'
BENCH_NUM_CHUNKS = 50

# Simulated RFC/ADR style chunks — realistic enough to extract 10–20 entities each
_BENCH_CHUNKS = [
    """\
RFC-ADR-{i}: Service-to-Service Authentication Design
Status: Accepted | Authors: Platform Engineering Team | Date: 2024-0{m}-15

Context: The DynamoDB Global Tables API requires mutual TLS authentication for
cross-region replication. The AWS IAM service controls authorization. The
Kong Gateway handles rate limiting. The Auth Service issues JWT tokens.

Decision: Use OAuth2 client credentials flow via the Auth Service, with token
caching via Redis. The API Gateway validates tokens before forwarding requests.
The Certificate Manager handles mTLS certificate rotation.

Consequences: The Token Validator must be deployed in each availability zone.
The Redis Cache requires high availability configuration. The Load Balancer
must support sticky sessions for WebSocket connections.
""",
    """\
RFC-ADR-{i}: Data Pipeline Observability Requirements
Status: Proposed | Authors: Data Platform Team | Date: 2024-0{m}-20

Context: The Kafka Consumer Group processes events from the Event Bus. The
ClickHouse Analytics database stores aggregated metrics. The Prometheus Metrics
Collector scrapes all service endpoints. The Grafana Dashboard visualizes
pipeline health. The PagerDuty integration handles alerting.

Decision: Implement OpenTelemetry distributed tracing across all pipeline
components. The Jaeger Backend stores trace data. The Tempo Service correlates
logs and traces. The SLO Tracker enforces latency budgets.

Consequences: The Data Exporter requires instrumentation updates. The Schema
Registry must expose health endpoints. The Flink Job Cluster needs trace context
propagation.
""",
    """\
RFC-ADR-{i}: Multi-Region Failover Strategy
Status: Accepted | Authors: Infrastructure Team | Date: 2024-0{m}-10

Context: The Route53 DNS Service manages global traffic routing. The CloudFront
Distribution handles edge caching. The S3 Bucket stores static assets with
cross-region replication. The RDS Aurora Global Database provides multi-region
reads. The Elastic Load Balancer distributes traffic across availability zones.

Decision: Implement active-active failover using Route53 health checks and
weighted routing policies. The Health Check Lambda monitors endpoint availability.
The SNS Topic triggers failover notifications. The CloudWatch Alarm detects
latency breaches.

Consequences: The Terraform Module for DNS must be updated. The Runbook
requires failover procedure documentation. The On-Call Engineer rotation needs
regional coverage.
""",
    """\
RFC-ADR-{i}: API Gateway Rate Limiting Policy
Status: Accepted | Authors: Security Engineering Team | Date: 2024-0{m}-05

Context: The Kong Gateway currently enforces no per-client rate limits. The
Abuse Detection Service flags suspicious traffic patterns. The API Key Manager
issues credentials to third-party developers. The Quota Service tracks usage.

Decision: Implement token bucket rate limiting at the Kong Gateway level.
The Redis Rate Limiter stores per-client counters. The Billing Service charges
for API calls above the free tier. The Developer Portal shows usage dashboards.

Consequences: The SDK Documentation must explain rate limit headers. The
Integration Test Suite needs rate limit simulation. The Client Library must
implement exponential backoff.
""",
    """\
RFC-ADR-{i}: Search Infrastructure Migration
Status: In Progress | Authors: Search Platform Team | Date: 2024-0{m}-25

Context: The Elasticsearch Cluster currently handles all full-text search.
The OpenSearch Service offers managed upgrades. The Vector Search Engine
enables semantic queries. The Document Ingestion Pipeline feeds search indexes.

Decision: Migrate from Elasticsearch to OpenSearch Service, adding a Vector
Search Engine alongside for hybrid retrieval. The Index Manager handles
schema migration. The Query Router directs queries to the appropriate engine.

Consequences: The Search API must support both query types during migration.
The Re-indexing Job requires a maintenance window. The Relevance Tuner needs
retraining on OpenSearch query syntax.
""",
]


def _make_chunk_content(chunk_index: int) -> str:
    template = _BENCH_CHUNKS[chunk_index % len(_BENCH_CHUNKS)]
    month = (chunk_index % 12) + 1
    return template.format(i=1000 + chunk_index, m=f'{month:02d}')


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.skip(f'Environment variable {name!r} is not set — skipping benchmark')
    return val


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_add_episode_latency_at_scale(tmp_path):
    """Measure add_episode wall time at 50K+ entity scale using LadybugDB.

    The test asserts that p95 latency is within ``BENCH_P95_TARGET_SECONDS``
    (60 s) for episodes using ``DeduplicationConfig(skip_llm_dedup=True)``.
    A second pass with ``DeduplicationConfig(cosine_min_score=0.85)`` is also
    measured and logged for comparison, but is not asserted (the SLO for full
    dedup depends on the underlying LadybugDB HNSW performance at scale).

    Skips if:
    - ``GRAPHITI_BENCH_LADYBUG_DB`` is not set in the environment, OR
    - The database contains fewer than ``BENCH_MIN_ENTITY_COUNT`` entities.
    """
    # -- Prerequisites -------------------------------------------------------
    pytest.importorskip(
        'real_ladybug',
        reason='real_ladybug package not installed — skipping benchmark',
    )
    db_path = _require_env('GRAPHITI_BENCH_LADYBUG_DB')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not openai_api_key and not anthropic_api_key:
        pytest.skip(
            'Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set — '
            'benchmark requires a real LLM client'
        )

    # -- Driver setup --------------------------------------------------------
    from graphiti_core.driver.ladybug_driver import LadybugDriver

    driver = LadybugDriver(db=db_path)

    # Count entities in the database (across all groups) to verify scale
    try:
        records, _, _ = await driver.execute_query(
            'MATCH (n:Entity) RETURN count(n) AS cnt'
        )
        entity_count = records[0]['cnt'] if records else 0
    except Exception as exc:
        pytest.skip(f'Could not count entities in {db_path!r}: {exc}')
        return

    if entity_count < BENCH_MIN_ENTITY_COUNT:
        pytest.skip(
            f'Database at {db_path!r} contains only {entity_count:,} entities '
            f'(need ≥ {BENCH_MIN_ENTITY_COUNT:,}) — populate the DB first'
        )

    logger.info('Benchmark: %d entities found in %s', entity_count, db_path)

    # -- LLM + embedder setup ------------------------------------------------
    if openai_api_key:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client import LLMConfig, OpenAIClient

        llm_client = OpenAIClient(LLMConfig(api_key=openai_api_key))
        embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=openai_api_key))
        logger.info('Benchmark: using OpenAI LLM + embedder')
    else:
        from graphiti_core.llm_client.anthropic_client import AnthropicClient
        from graphiti_core.llm_client.config import LLMConfig

        # Embedder still requires OpenAI for text-embedding-* models;
        # if only Anthropic key is present, use a voyageai embedder if available.
        try:
            voyage_key = os.environ.get('VOYAGE_API_KEY', '')
            from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig

            embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key=voyage_key))
            logger.info('Benchmark: using VoyageAI embedder')
        except Exception:
            pytest.skip(
                'Anthropic key set but no embedder available '
                '(need OPENAI_API_KEY or VOYAGE_API_KEY)'
            )
            return

        llm_client = AnthropicClient(LLMConfig(api_key=anthropic_api_key))
        logger.info('Benchmark: using Anthropic LLM')

    # -- Graphiti setup ------------------------------------------------------
    from graphiti_core import DeduplicationConfig, Graphiti
    from graphiti_core.nodes import EpisodeType

    graphiti = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)
    await graphiti.build_indices_and_constraints()

    # -- Benchmark run -------------------------------------------------------
    async def _run_pass(
        label: str,
        dedup_config: DeduplicationConfig | None,
        num_chunks: int = BENCH_NUM_CHUNKS,
    ) -> list[float]:
        latencies: list[float] = []
        ref_time = datetime.now(tz=timezone.utc)

        for i in range(num_chunks):
            content = _make_chunk_content(i)
            t0 = time.monotonic()
            await graphiti.add_episode(
                name=f'bench-chunk-{i:03d}',
                episode_body=content,
                source_description='Benchmark RFC/ADR document chunk',
                reference_time=ref_time,
                source=EpisodeType.text,
                group_id=BENCH_GROUP_ID,
                dedup_config=dedup_config,
            )
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            logger.info(
                '%s chunk %d/%d: %.2f s',
                label,
                i + 1,
                num_chunks,
                elapsed,
            )

        return latencies

    # Pass 1: skip_llm_dedup=True — fastest; this is the SLO-asserted path
    skip_config = DeduplicationConfig(skip_llm_dedup=True)
    latencies_skip = await _run_pass('skip_llm_dedup', skip_config)

    p50_skip = statistics.median(latencies_skip)
    p95_skip = statistics.quantiles(latencies_skip, n=20)[18]  # 95th percentile
    p99_skip = max(latencies_skip)

    logger.info(
        'BENCH skip_llm_dedup=True | n=%d | p50=%.2fs | p95=%.2fs | p99=%.2fs',
        len(latencies_skip),
        p50_skip,
        p95_skip,
        p99_skip,
    )

    # Pass 2: cosine_min_score=0.85 — threshold-based; logged but not asserted
    threshold_config = DeduplicationConfig(cosine_min_score=0.85)
    latencies_threshold = await _run_pass('cosine_min_score=0.85', threshold_config)

    p95_threshold = statistics.quantiles(latencies_threshold, n=20)[18]
    logger.info(
        'BENCH cosine_min_score=0.85 | n=%d | p50=%.2fs | p95=%.2fs | p99=%.2fs',
        len(latencies_threshold),
        statistics.median(latencies_threshold),
        p95_threshold,
        max(latencies_threshold),
    )

    # -- SLO assertion (skip_llm_dedup path only) ----------------------------
    assert p95_skip <= BENCH_P95_TARGET_SECONDS, (
        f'SLO BREACH: skip_llm_dedup p95={p95_skip:.2f}s exceeds '
        f'{BENCH_P95_TARGET_SECONDS}s target at {entity_count:,} entities. '
        f'Latencies: {[f"{lat:.2f}" for lat in latencies_skip]}'
    )

    logger.info(
        'BENCH PASS: skip_llm_dedup p95=%.2fs ≤ %.0fs SLO at %d entities',
        p95_skip,
        BENCH_P95_TARGET_SECONDS,
        entity_count,
    )
