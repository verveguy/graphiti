# ADR 005: LadybugDB Connection Health Probe on Exception

**Status**: Accepted  
**Date**: 2026-05-04  
**Related**: verveguy/graphiti#74

## Context

`real_ladybug` (the PyPI package wrapping a community fork of KuzuDB) surfaces
C++ `std::out_of_range` exceptions — such as `unordered_map::at: key not found`
— as standard Python `RuntimeError` instances. Python cannot distinguish these
native C++ exceptions from ordinary Python-level `RuntimeError` by type alone;
only the exception message string identifies the origin.

During large-scale ingestion (~40K+ entities), these exceptions were observed
mid-chunk, typically correlated with HNSW vector index operations in
`hnsw_safe_writes.py`. In four confirmed cases the caught exception was
recoverable — the `LadybugDriver` instance remained usable and subsequent docs
ingested successfully. In one case the error escalated to SIGSEGV, corrupting
the database WAL file.

The risk profile is asymmetric: a silently corrupted connection that is not
detected will cause the next HNSW write to potentially SIGSEGV, killing the
process and corrupting the database. Detecting the corruption early and failing
loudly is strongly preferable.

## Decision

After **any** exception raised by `LadybugDriver.execute_query`, immediately
probe the connection by executing `RETURN 1` on the same `AsyncConnection`.

- If the probe succeeds: the connection is healthy. Re-raise the original
  exception unchanged. The caller can catch it, skip the chunk, and continue.
- If the probe fails: the connection is poisoned. Log at `CRITICAL` level and
  raise `LadybugConnectionError` (a new `GraphitiError` subclass defined in
  `graphiti_core/errors.py`). Callers must treat the driver session as unusable
  and not issue further queries.

The probe runs after **all** exceptions, not just those matching `unordered_map`
in the message string. Reasons:

1. **Future-proofing**: KuzuDB C++ exception messages are internal implementation
   details and may change across versions. A string-match guard creates a
   maintenance burden and would silently miss new exception variants.
2. **Negligible overhead**: One lightweight `RETURN 1` round-trip on the error
   path, which is already failing. The common (non-error) path has zero overhead.
3. **Simplicity**: A single consistent policy is easier to reason about than a
   conditional probe.

The `unordered_map` substring match is retained solely for log messaging — to
distinguish a known native C++ exception from a generic Python error in the log
output — and does not gate the probe.

## Consequences

- **`LadybugConnectionError`** is raised from `execute_query` only when the
  connection health probe itself fails. Callers that previously caught `Exception`
  broadly will continue to work; callers that need to distinguish a poisoned
  connection can now `except LadybugConnectionError`.
- **SIGSEGV residual risk**: If KuzuDB's internal HNSW index state is silently
  corrupted after a caught exception, the `RETURN 1` probe may succeed (it does
  not exercise the HNSW path) while the next HNSW write still SIGSEGVs. The
  probe guards against an obviously broken connection, not against silent internal
  state corruption. This risk is documented and accepted; a deeper fix would
  require upstream KuzuDB changes.
- **ADR 004 threading boundary**: The health probe uses `self.client.execute()`
  (the `AsyncConnection`), which is event-loop-bound. This is correct: the probe
  only runs in `execute_query`, which is always called from the async path. The
  WAL replay path uses a separate synchronous `ladybug.Connection` and is
  unaffected.
- **No user-facing changes**: The probe is entirely internal to the driver layer.
  No new configuration, environment variables, or CLI flags are introduced.
