# ADR 001: merge_entities — no explicit database transactions

**Status:** Accepted
**Date:** 2026-04-30
**Issue:** #54 — Add merge_entities operation for cleaning up duplicates in existing graphs

## Context

The `merge_entities` operation performs a multi-step write sequence for each duplicate node:

1. Rewire entity edge endpoints (fetch → modify `source_node_uuid` / `target_node_uuid` → save each edge)
2. Deduplicate colliding edges (delete)
3. Rewire episodic MENTIONS edges (fetch → update `target_node_uuid` → save or delete)
4. Save updated survivor (union of labels, appended summary)
5. Delete duplicate node (`DETACH DELETE`)

Each step is a separate database round-trip. If the process crashes between steps, the graph is left in a partially-merged state.

The question is: should these steps be wrapped in a single explicit database transaction?

## Decision

**No explicit transactions.** The orchestration in `Graphiti.merge_entities()` issues each step as a separate auto-committed query, matching the existing pattern of all other multi-step `Graphiti` public methods (`add_episode`, `remove_episode`, `add_triplet`, etc.).

## Rationale

### 1. FalkorDB, LadybugDB, and Neptune do not support real multi-statement transactions

All three backends use a `_SessionTransaction` shim (see `graphiti_core/driver/driver.py`) that executes queries immediately rather than buffering them in a real transaction. Wrapping the merge in a `driver.transaction()` context on these backends provides no atomicity guarantee — it is a no-op from a consistency standpoint.

### 2. No existing precedent for explicit transactions in Graphiti public methods

None of `add_episode`, `remove_episode`, `add_triplet`, or `search_` wrap multi-step writes in explicit transactions. Introducing explicit transactions only for `merge_entities` would create an inconsistent API surface: callers of Neo4j would get atomicity guarantees that callers of FalkorDB/LadybugDB/Neptune do not.

### 3. merge_entities is a cleanup operation, not a hot path

Merge is invoked manually (or via MCP) for legacy graph cleanup, not on every ingestion request. The consequence of partial failure (some edges rewired but duplicate node not yet deleted) is a temporarily inconsistent graph, not data loss. Re-running `merge_entities` with the same inputs is idempotent: already-rewired edges simply don't match the duplicate anymore, and the operation completes cleanly.

### 4. Idempotency mitigates partial failure risk

If `merge_entities` is interrupted mid-run, the caller can safely re-invoke it. The operation fetches the current state of edges and MENTIONS at runtime and only rewires edges that still point to the duplicate. Edges already rewired in a prior run will not be re-processed.

## Consequences

- **On Neo4j**: merge is not atomic. A crash between steps leaves the graph in a partially-merged state. Re-running the operation recovers cleanly.
- **On FalkorDB / LadybugDB / Neptune**: same behaviour — no atomicity guarantee exists regardless.
- **Callers must treat merge as best-effort atomic** and be prepared to re-run it in the event of failure.
- **Docstring documents this**: `Graphiti.merge_entities()` explicitly states the best-effort atomicity guarantee in its docstring.

## Alternatives considered

### Wrap in `driver.transaction()` on Neo4j only

Would provide atomicity on Neo4j but create a different execution model per backend (transactional on Neo4j, non-transactional elsewhere). The divergence would be invisible to callers and difficult to test. Rejected for consistency.

### Compensating restore on partial failure

If step N fails, undo steps 1…N-1. This is complex to implement correctly (requires saving pre-merge state), introduces new failure modes, and still doesn't help on backends without real transactions. Rejected as over-engineering for a cleanup operation.
