# ADR 005 — DeduplicationConfig: configurable dedup intensity for bulk-seed workloads

**Date:** 2026-05-05
**Status:** Accepted

## Context

At 50K–80K+ entities in a single `group_id`, individual `add_episode` calls can exceed 5 minutes
of wall-clock time. The root cause is a non-linear cost curve in the node deduplication pipeline:

1. **HNSW + BM25 query latency** grows with graph cardinality. On LadybugDB (a KuzuDB-derived
   embedded graph), the HNSW DELETE→CREATE write pattern (required because HNSW-indexed columns
   cannot be updated in place) adds per-entity write cost that compounds at large scale.
2. **Candidate pool ambiguity** grows with the graph. With `NODE_DEDUP_COSINE_MIN_SCORE = 0.3`,
   more entities fall in the 0.3–0.7 cosine range as the graph grows. These are escalated to the
   LLM.
3. **LLM prompt size** scales with ambiguous candidates. With 15 unresolved nodes × 10 candidates
   each, the prompt can reach 150 candidate entries — slow to generate even with Anthropic prompt
   caching.

Research confirmed that HNSW/BM25 parallelism across extracted nodes (via `semaphore_gather`) and
LLM batching (single call per `add_episode`) are **already implemented**. The cliff is not a missing
optimization — it is a scaling property of the underlying database and candidate pool growth.

The lever is reducing the work the dedup pipeline does per episode at large scale.

### Design options considered

**Option A — DeduplicationConfig dataclass:**
A single `DeduplicationConfig(skip_llm_dedup=True, cosine_min_score=0.85)` object passed to
`add_episode`. Callers import one type; the dataclass is extensible without signature churn.

**Option B — Individual keyword arguments on `add_episode`:**
`add_episode(..., skip_llm_dedup=True, cosine_min_score=0.85)`. Simpler to land; no new type.

**Option C — Enum `dedup_mode` on `add_episode`:**
`add_episode(..., dedup_mode=DedupMode.SKIP_LLM)`. Finite set of named modes; easier to document.

### Threshold semantics: "3-bucket" vs "raise the floor"

Within the `cosine_min_score` dimension, two approaches were considered:

**"3-bucket"** — retrieve all candidates, then:
  - Auto-resolve candidates with cosine > threshold as duplicates (no LLM)
  - Auto-reject candidates with cosine < inverse threshold (no LLM)
  - Send ambiguous candidates to LLM

This requires propagating per-candidate cosine scores through
`_semantic_candidate_search → _merge_candidate_nodes → _build_candidate_indexes`.
The current return type is `list[EntityNode]` with no score attached. Adding scores
would touch 5+ internal functions and change the internal data model.

**"Raise the floor"** — set `cosine_min_score=0.85` to exclude low-scoring entities from the
HNSW retrieval entirely. Fewer candidates return, fewer nodes escalate to LLM, and the LLM prompt
shrinks. No internal restructuring is needed.

## Decision

**Option A (DeduplicationConfig dataclass) was chosen.**

Two fields are implemented:
- `skip_llm_dedup: bool = False` — gates the `_resolve_with_llm` call entirely
- `cosine_min_score: float = NODE_DEDUP_COSINE_MIN_SCORE` — HNSW retrieval floor
- `candidate_limit: int = NODE_DEDUP_CANDIDATE_LIMIT` — max candidates per node

**"Raise the floor"** was chosen over "3-bucket" for `cosine_min_score`.

## Rationale

### Why a dataclass over individual params

`add_episode` is a public API with an already-long signature. Adding two unrelated bool/float
params (`skip_llm_dedup`, `cosine_min_score`) would make the signature harder to read and harder
to extend. A dataclass groups the semantically related fields, gives them a discoverable import
path (`from graphiti_core import DeduplicationConfig`), and can absorb future fields (e.g., a
`candidate_limit` override, a `dedup_timeout` budget) without further signature changes.

Option C (enum) was rejected because the two modes are orthogonal — a caller might want
`skip_llm_dedup=True` with default cosine floor, or raised cosine floor with LLM still enabled.
An enum would require a combinatorial explosion of named modes (`SKIP_LLM`, `HIGH_THRESHOLD`,
`SKIP_LLM_HIGH_THRESHOLD`, etc.) or lose expressiveness.

### Why "raise the floor" over "3-bucket"

The "3-bucket" approach is conceptually cleaner — it would auto-resolve near-certain matches
without LLM — but it requires:

1. Changing `_semantic_candidate_search` to return `list[tuple[EntityNode, float]]` instead of
   `list[EntityNode]`.
2. Updating `_merge_candidate_nodes` to preserve scores through deduplication.
3. Updating `_build_candidate_indexes` to accept and propagate scores.
4. Adding a new resolution path in `resolve_extracted_nodes` between "similarity" and "LLM".

This touches 5+ internal functions and changes the established internal data model. The "raise the
floor" approach achieves the same practical outcome for the target corpus (RFC/ADR): by setting
`cosine_min_score=0.85`, ambiguous low-similarity candidates are excluded from the pool, which
both shrinks the LLM prompt and reduces the number of nodes escalated to LLM. The trade-off is
symmetrical: entities whose true duplicate scores below 0.85 are missed entirely — but for
corpora with distinctive entity names, this is acceptable.

A future ADR may revisit "3-bucket" if the internal scoring data model is changed for other
reasons (e.g., cross-encoder reranking of candidates).

### Why the SLO is `skip_llm_dedup` only

The 60-second p95 SLO (R2) is asserted by the benchmark for the `skip_llm_dedup=True` path only.
With full LLM dedup at 80K+ entities, the SLO may not be achievable — the LadybugDB HNSW write
overhead (DELETE→CREATE per entity) is a floor that application-level changes cannot fully address.
The `cosine_min_score=0.85` path is benchmarked and logged but not asserted. Deeper LadybugDB-side
fixes are tracked separately (sister issue: `unordered_map::at` errors at 80K+ entities).

## Consequences

### For callers

- `add_episode` gains an optional `dedup_config: DeduplicationConfig | None = None` parameter.
  Existing call sites are unaffected (default is `None`, which preserves current behavior).
- `DeduplicationConfig` is re-exported from `graphiti_core` for a clean import path.
- `dedupe_nodes_bulk` in `bulk_utils.py` gains the same optional parameter, enabling bulk paths
  to be accelerated without requiring callers to thread the config through manually.

### For contributors

- `_semantic_candidate_search` now accepts `cosine_min_score` and `candidate_limit` as explicit
  parameters (defaulting to the module-level constants). The constants remain as canonical defaults.
- Future additions to dedup configuration belong in `DeduplicationConfig`, not as new keyword
  arguments on `add_episode`.
- The "raise the floor" semantics for `cosine_min_score` must be documented clearly: raising the
  value **excludes** low-similarity entities from HNSW retrieval, not from LLM consideration.
  A future contributor who assumes the threshold auto-resolves high-cosine candidates as duplicates
  would implement incorrect behavior.

### Dedup quality trade-off

| Config | Dedup quality | Latency at 80K entities |
|--------|---------------|------------------------|
| Default (`None`) | Highest | May exceed 5 min |
| `cosine_min_score=0.85` | Misses duplicates with cosine < 0.85 | Reduced |
| `skip_llm_dedup=True` | Deterministic only (exact + MinHash/LSH) | p95 ≤ 60 s (target) |

Post-hoc dedup cleanup is available via `merge_entities` and `merge_entities_batch`.
