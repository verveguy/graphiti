# ADR 004: WAL Replay Threading Boundary

**Status**: Accepted  
**Date**: 2026-05-04  
**Related**: verveguy/graphiti#48

## Context

`replay_wal_ladybug()` replays JSONL WAL files into a LadybugDB database.
The body is synchronous: JSONL parsing, `BEGIN TRANSACTION`, repeated
`conn.execute()` calls, and `COMMIT`. On production workspaces this takes
20+ minutes.

Because `replay_wal_ladybug()` is an `async def` called from an asyncio event
loop, this synchronous work completely starves the event loop. No other
coroutines can run, including the progress-drain loop in the calling service
layer that streams `notifications/progress` events to clients.

A partial fix (Option B) added `asyncio.sleep(0)` yields at batch boundaries
and between WAL files. These reduced burst length but did not solve the
problem: within a single large batch (up to ~30 seconds of CPU work), the
event loop was still blocked.

## Decision

Move the synchronous replay body off the asyncio event loop using
`asyncio.to_thread()`.

A new module-level function `_sync_replay_wal_ladybug()` contains the pure
synchronous body (JSONL parsing, `ladybug.Connection` creation, batch loop,
binary-split retry, `conn.close()`). The public `replay_wal_ladybug()` async
function dispatches to it via:

```python
replayed, skipped, errors = await asyncio.to_thread(
    _sync_replay_wal_ladybug,
    driver.db if driver is not None else None,
    wal_files,
    from_seq,
    dry_run,
    batch_size,
)
```

The event loop is then free to schedule the progress-drain coroutine and
stream events in real time throughout the replay.

### Threading boundary: why `driver.db`, not `LadybugDriver`

`LadybugDriver.__init__()` creates a `kuzu.AsyncConnection`. The internals of
`AsyncConnection.__init__()` may capture the running asyncio event loop or
post work onto it. Creating a `LadybugDriver` from a worker thread that is not
the event loop thread is therefore unsafe and can cause silent failures or
wrong-loop errors.

The safe split:

| What | Where |
|------|-------|
| `LadybugDriver(db=db)` construction | Event loop (before `asyncio.to_thread()`) |
| `driver.db` (a `kuzu.Database` handle) | Passed into worker thread as argument |
| `ladybug.Connection(driver.db)` creation | Worker thread (sync, for batch transactions) |
| `driver.build_indices_and_constraints()` | Event loop (after thread completes) |
| `driver.close()` | Event loop (in `finally`, always runs) |

`ladybug.Connection` is documented as not thread-safe for concurrent access,
but since the replay thread is the only consumer of `conn` during replay (it
is a local variable, not shared), this is safe.

### Exception propagation

`asyncio.to_thread()` re-raises worker thread exceptions on the event loop
transparently. `conn.close()` is in `_sync_replay_wal_ladybug()`'s own
`try/finally` and always runs on the worker thread. `driver.close()` is in
the outer async function's `finally` and always runs on the event loop, even
if the thread raises.

### Option B yields removed

The `asyncio.sleep(0)` calls from Option B are removed. Running in a thread
means the event loop is always free; cooperative yielding within the sync body
is pointless.

## Cross-repo dependency: WalProgressHandler thread-safety

The `WalProgressHandler` lives in `liminis-framework`. Its `emit()` method
calls `asyncio.Queue.put_nowait()` synchronously. Once replay runs in a worker
thread, `logger.info()` calls inside `_sync_replay_wal_ladybug()` invoke
`emit()` from the worker thread — which makes `put_nowait()` on an asyncio
queue unsafe (not thread-safe from outside the event loop thread).

The fix in `liminis-framework` is to capture `asyncio.get_running_loop()` at
handler creation and replace `queue.put_nowait(item)` with
`loop.call_soon_threadsafe(queue.put_nowait, item)` in `emit()`. This work is
tracked in liminis-framework#142 and #148.

**Impact**: After this graphiti-core PR ships, the event loop is unblocked and
the drain coroutine can run. However, progress events will not be correctly
delivered until `liminis-framework` is also updated. The two PRs together
complete live streaming.

## Prior art

`asyncio.to_thread()` is already used in this codebase in
`graphiti_core/llm_client/gliner2_client.py` to offload synchronous GLiNER2
model inference. The same pattern applies here.

## Consequences

- The event loop is free throughout WAL replay; progress-drain coroutines and
  other concurrent async work are no longer starved.
- `_sync_replay_wal_ladybug()` is a module-level function (not a nested
  closure) to keep the callable simple and testable.
- The threading boundary (create `LadybugDriver` on the event loop; pass only
  `driver.db` into the thread) is non-obvious and not encoded in types. This
  ADR documents it so future contributors do not accidentally move
  `LadybugDriver` construction into the thread.
