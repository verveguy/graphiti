## Context

The graphiti knowledge graph database (`.graphiti/db`) is a 149MB FalkorDB RDB binary that's too large for GitHub. It's also expensive to reconstruct — each episode requires LLM extraction calls. Workspace git repos using graphiti end up unable to push because of the large binary file.

**Goal**: Capture every graph mutation (post-LLM, pre-FalkorDB) as a replayable write-ahead log. This enables:
- **Git-friendly backup** — small text files instead of 149MB binary
- **Full DB reconstruction** by replaying Cypher mutations — no LLM re-extraction needed
- `.graphiti/db` can be gitignored; the WAL becomes the source of truth for backup

## Interception Point

ALL FalkorDB mutations flow through exactly **two methods** in one file:

**`graphiti_core/driver/falkordb_driver.py`**
- `FalkorDriver.execute_query()` (line 229) — non-transactional path
- `FalkorDriverSession.run()` (lines 100, 104) — transactional path

Both call `await graph.query(cypher, params)`. We log after successful execution.

## Implementation Plan

### Step 1: Create `graphiti_core/driver/wal.py`

New module — pure Python I/O, no FalkorDB dependency.

```python
class WalWriter:
    def __init__(self, wal_dir: str | Path, max_events_per_file: int = 10_000)
    async def log_mutation(self, cypher: str, params: dict, database: str) -> None
    async def close(self) -> None

    @staticmethod
    def is_mutation(cypher: str) -> bool       # detect CREATE/MERGE/SET/DELETE/DETACH/DROP/REMOVE
    @staticmethod
    def is_index_ddl(cypher: str) -> bool      # detect CREATE INDEX, CREATE FULLTEXT INDEX, etc.
```

**Write detection**: Check for mutation keywords (CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE) after stripping string literals. Intentionally permissive — false positive wastes bytes, false negative loses data.

**Index exclusion**: `is_index_ddl()` filters out CREATE INDEX / CREATE FULLTEXT INDEX / CREATE VECTOR INDEX / DROP INDEX queries. These are excluded because replay calls `build_indices_and_constraints()` separately and they're idempotent.

**File naming**: `{timestamp}_{session_uuid_short}_{file_seq:04d}.jsonl`
Example: `20260323T180000Z_a1b2c3_0001.jsonl`

**JSONL format** (one line per mutation):
```json
{"seq": 42, "ts": "2026-03-23T18:00:01.234Z", "db": "graphiti_memory", "cypher": "MERGE (n:Entity {uuid: $uuid}) SET n.name = $name", "params": {"uuid": "abc-123", "name": "Alice"}}
```

**Storage as directory of files** (not single growing file) for git efficiency:
- Already-committed files are never modified — zero redundancy in git
- Each commit only adds new small file blobs
- Pushes are tiny — just the new files
- New file on service start + after `max_events_per_file`; completed files are immutable

**Sequence continuity**: On startup, scan existing WAL files for highest `seq` and continue from there.

**Concurrency**: `asyncio.Lock` to prevent interleaved JSONL lines from concurrent tasks.

### Step 2: Modify `FalkorDriver.__init__()` — add WAL params

```python
def __init__(self, ..., wal_dir: str | Path | None = None, wal_writer: WalWriter | None = None):
```

- `wal_dir` provided, no `wal_writer` → create new `WalWriter(wal_dir)`
- `wal_writer` provided (from `clone()`) → reuse it
- Neither → WAL disabled (`self._wal = None`)

### Step 3: Instrument `FalkorDriver.execute_query()`

Log **after** successful `graph.query()` (only actually-executed mutations are logged):

```python
result = await graph.query(cypher_query_, params)
# WAL: log if mutation and not index DDL
if self._wal is not None:
    await self._wal.log_mutation(cypher_query_, params, database=self._database)
```

The `log_mutation` method internally calls `is_mutation()` / `is_index_ddl()` and skips non-qualifying queries.

### Step 4: Instrument `FalkorDriverSession`

Add WAL + database to session init:

```python
class FalkorDriverSession(GraphDriverSession):
    def __init__(self, graph: FalkorGraph, wal: WalWriter | None = None, database: str | None = None):
```

Update `FalkorDriver.session()` to pass WAL and database name through.

Instrument `run()` — same pattern as `execute_query`, log after each successful `graph.query()`.

### Step 5: Update `clone()` to share WAL

```python
def clone(self, database: str) -> 'GraphDriver':
    ...
    cloned = FalkorDriver(falkor_db=self.client, database=database, wal_writer=self._wal)
```

All cloned drivers write to the same WAL with their respective `database` field.

### Step 6: Update `close()` to flush WAL

```python
async def close(self) -> None:
    if self._wal is not None:
        await self._wal.close()
    # ... existing close logic ...
```

### Step 7: Wire into `graphiti_service.py` (liminis-framework)

> **Note**: This step lives in `verveguy/liminis-framework`, not this repo.

At line ~462 in `framework/.claude/skills/graphiti/scripts/graphiti_service.py` where `FalkorDriver` is constructed:

```python
wal_dir = str(Path(workspace_root) / ".graphiti" / "wal")
driver = FalkorDriver(host="127.0.0.1", port=embedded_port, database=db_name, wal_dir=wal_dir)
```

### Step 8: Create replay script `graphiti_core/driver/wal_replay.py`

```python
async def replay_wal(wal_dir, host, port, database, from_seq=0, dry_run=False) -> int:
```

1. Connect to fresh FalkorDB (WAL disabled on the replay driver)
2. Call `build_indices_and_constraints()` first
3. Read all `.jsonl` files sorted by filename
4. Execute each `(cypher, params)` against the `db` specified in the event
5. Skip events with `seq < from_seq` (for partial replay / resume)
6. CLI entry point via `if __name__ == '__main__'` with argparse

### Step 9: Update workspace `.gitignore`

Add:
```
.graphiti/db
.graphiti/db.settings
```

The WAL directory (`.graphiti/wal/`) stays tracked.

## Files Summary

| File | Action |
|------|--------|
| `graphiti_core/driver/wal.py` | **Create** — WalWriter class |
| `graphiti_core/driver/wal_replay.py` | **Create** — replay script + CLI |
| `graphiti_core/driver/falkordb_driver.py` | **Modify** — add wal_dir/wal_writer params, instrument execute_query, session, clone, close |
| `tests/driver/test_wal.py` | **Create** — unit + integration tests |
| _(liminis-framework)_ `graphiti_service.py` | **Modify** — pass wal_dir when constructing FalkorDriver |

## Known Considerations

- **Embedding vectors in params**: Some mutations include 768-dim float vectors (~15KB per line). At scale this is manageable — git packfiles delta-compress well, and individual WAL files are much smaller than the 149MB RDB.
- **Non-deterministic replay**: Replayed DB won't be bit-identical to original (FalkorDB internal layout may differ), but semantically equivalent.
- **`with_database()` shallow copy**: Uses `copy.copy(self)` which shares `_wal` reference automatically — correct behavior.

## Verification

1. **Unit tests**: `is_mutation()` correctly classifies reads vs writes; `is_index_ddl()` catches all index patterns; file rotation works; sequence numbers are monotonic and survive restarts
2. **Integration test**: Start graphiti service with WAL → index a test document → verify WAL files appear with expected mutations → replay into fresh FalkorDB → query both DBs and compare results
3. **Git test**: Commit WAL files → index more → commit again → verify only new WAL files in diff, old ones unchanged

---
<details>
<summary><h2>Research Findings</h2></summary>

_Research conducted: 2026-03-24_

### Summary

The implementation plan is validated by codebase analysis. All mutation paths are correctly identified, and the proposed WAL integration points are accurate. Key findings include specific mutation patterns, serialization details, and clone/session behavior that inform implementation.

### Key Discoveries

- **Mutation interception points confirmed**: `FalkorDriver.execute_query()` at line 222-253 and `FalkorDriverSession.run()` at lines 95-106 are the only paths to FalkorDB (`falkordb_driver.py`)
- **8 distinct mutation pattern categories** found across the codebase (MERGE/SET for nodes/edges, DELETE, DETACH DELETE, DROP INDEX, CREATE INDEX, bulk UNWIND operations)
- **Embedding dimensions configurable**: Default 1024, but varies by provider (Gemini defaults to 768) — affects WAL line size estimates
- **`clone()` creates new instance**: Unlike `with_database()` which uses `copy.copy()`, `clone()` creates a fresh `FalkorDriver` — WAL must be passed via `wal_writer` parameter explicitly
- **Datetime serialization already handled**: `convert_datetimes_to_strings()` at `utils/datetime_utils.py:45-55` converts all datetime objects to ISO 8601 strings before params reach the driver
- **Two distinct multiple-driver patterns found**: (1) Instance replacement pattern in `graphiti.py:886-890` where `self.driver` is replaced when switching group_ids, (2) Temporary concurrent clones via `@handle_multiple_group_ids` decorator in `decorators.py:47-68` for parallel multi-database operations

### Detailed Findings

#### Mutation Patterns Found

All mutations flow through the two identified methods. Categories discovered:

| Pattern | Example Location | Sample Query |
|---------|------------------|--------------|
| MERGE + SET (nodes) | `models/nodes/node_db_queries.py:131-138` | `MERGE (n:Entity {uuid: $uuid}) SET n = $entity_data` |
| MERGE + SET (edges) | `models/edges/edge_db_queries.py:65-73` | `MERGE (source)-[e:RELATES_TO {uuid: $uuid}]->(target) SET e = $edge_data` |
| DETACH DELETE | `driver/falkordb/operations/entity_node_ops.py:100-107` | `MATCH (n {uuid: $uuid}) DETACH DELETE n` |
| DELETE (edge only) | `driver/falkordb/operations/entity_edge_ops.py:104-107` | `MATCH ()-[e {uuid: $uuid}]->() DELETE e` |
| Bulk UNWIND | `models/nodes/node_db_queries.py:193-201` | `UNWIND $nodes AS node MERGE (n:Entity {uuid: node.uuid})...` |
| DROP INDEX | `driver/falkordb/operations/graph_ops.py:90-102` | `DROP INDEX ON :Entity(uuid)` |
| CREATE INDEX | `graph_queries.py:29-49` | `CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id...)` |
| Clear all | `driver/falkordb/operations/graph_ops.py:43` | `MATCH (n) DETACH DELETE n` |

#### `is_mutation()` Implementation Guidance

Based on patterns found, mutation detection should check for these keywords (case-insensitive, after stripping string literals):
- `CREATE` (but exclude `CREATE INDEX`, `CREATE FULLTEXT INDEX`)
- `MERGE`
- `SET`
- `DELETE`
- `DETACH`
- `DROP` (but exclude `DROP INDEX`)
- `REMOVE`

**False positive risk**: `CALL db.idx.fulltext.createNodeIndex(...)` contains "create" but is index DDL — the `is_index_ddl()` check handles this.

#### `is_index_ddl()` Patterns to Detect

From `graph_queries.py` and `driver/falkordb/operations/graph_ops.py`:
- `CREATE INDEX FOR`
- `CREATE FULLTEXT INDEX`
- `DROP INDEX ON`
- `DROP FULLTEXT INDEX`
- `CALL db.idx.fulltext.createNodeIndex`
- `CALL db.idx.fulltext.queryNodes` (read-only, but good to exclude)

#### Serialization Details for WAL

Parameters are already serialized before reaching the driver:

1. **Datetime objects** → ISO 8601 strings via `convert_datetimes_to_strings()` (`utils/datetime_utils.py:45-55`)
2. **UUIDs** → strings (Pydantic serializes them)
3. **Embedding vectors** → `list[float]` passed as-is; FalkorDB converts via `vecf32()` in Cypher

**WAL serialization**: Standard `json.dumps()` should work for all param types after datetime conversion. No special handling needed.

**Line size estimate**: Entity with 1024-dim embedding ≈ 15-20KB per mutation line. Community mutations without embeddings ≈ 0.5-1KB.

#### Clone and Session Behavior

**`clone()` at `falkordb_driver.py:306-319`**:
```python
def clone(self, database: str) -> 'GraphDriver':
    if database == self._database:
        cloned = self
    elif database == self.default_group_id:
        cloned = FalkorDriver(falkor_db=self.client)  # No WAL passed!
    else:
        cloned = FalkorDriver(falkor_db=self.client, database=database)  # No WAL passed!
    return cloned
```
**Issue**: Current `clone()` doesn't pass WAL. Implementation must add `wal_writer=self._wal` parameter.

**`with_database()` at `driver.py:117-125`**:
```python
def with_database(self, database: str) -> GraphDriver:
    cloned = copy.copy(self)  # Shallow copy shares _wal reference
    cloned._database = database
    return cloned
```
**Correct**: Shallow copy automatically shares `_wal` instance — no changes needed.

**`session()` at `falkordb_driver.py:255-256`**:
```python
def session(self, database: str | None = None) -> GraphDriverSession:
    return FalkorDriverSession(self._get_graph(database))
```
**Issue**: Doesn't pass WAL or database name. Must be updated to:
```python
return FalkorDriverSession(self._get_graph(database), wal=self._wal, database=database or self._database)
```

#### Multiple Driver Usage Patterns

**Pattern 1: Instance Replacement** (`graphiti.py:886-890`, `graphiti.py:1112-1116`):
```python
if group_id != self.driver._database:
    self.driver = self.driver.clone(database=group_id)
    self.clients.driver = self.driver
```
- Driver is replaced on the Graphiti instance when switching group_ids
- Both `self.driver` and `self.clients.driver` are updated
- Clone becomes the "primary" driver for the instance

**Pattern 2: Temporary Concurrent Clones** (`decorators.py:47-68`):
```python
async def execute_for_group(gid: str):
    return await func(
        self,
        *filtered_args,
        **{**kwargs, 'group_ids': [gid], 'driver': driver.clone(database=gid)},
    )

results = await semaphore_gather(
    *[execute_for_group(gid) for gid in group_ids],
    max_coroutines=getattr(self, 'max_coroutines', None),
)
```
- Temporary clones created for parallel execution across multiple databases
- Used by `@handle_multiple_group_ids` decorator
- Clones are short-lived and not persisted on the instance
- FalkorDB-specific multi-tenancy feature

**WAL Implication**: Since both patterns exist, `close()` cannot rely on "who created the WAL" — temporary clones never call `close()`. Recommendation: Make `WalWriter.close()` idempotent (first call flushes and closes, subsequent calls are no-ops).

#### Test Patterns

Existing test patterns in `tests/driver/test_falkordb_driver.py`:
- Uses `@pytest.mark.asyncio` for async tests
- Uses `unittest.mock.MagicMock` and `AsyncMock` for mocking
- Uses `@unittest.skipIf(not HAS_FALKORDB, ...)` for optional FalkorDB tests
- Integration tests use `pytest.importorskip('falkordb')`

WAL tests should follow same patterns with:
- Unit tests for `is_mutation()`, `is_index_ddl()`, file rotation, sequence continuity
- Mock-based tests for WAL integration in driver
- Integration tests requiring temp directory for WAL files

### Files Requiring Changes

| File | Lines | Change |
|------|-------|--------|
| `graphiti_core/driver/wal.py` | N/A (new) | Create WalWriter class with `log_mutation()`, `is_mutation()`, `is_index_ddl()`, file rotation, sequence tracking |
| `graphiti_core/driver/wal_replay.py` | N/A (new) | Create replay function + CLI with argparse |
| `graphiti_core/driver/falkordb_driver.py` | 115-168 | Add `wal_dir` and `wal_writer` params to `__init__()` |
| `graphiti_core/driver/falkordb_driver.py` | 222-236 | Instrument `execute_query()` to call `log_mutation()` after successful query |
| `graphiti_core/driver/falkordb_driver.py` | 74-77 | Add `wal` and `database` params to `FalkorDriverSession.__init__()` |
| `graphiti_core/driver/falkordb_driver.py` | 95-106 | Instrument `FalkorDriverSession.run()` to call `log_mutation()` |
| `graphiti_core/driver/falkordb_driver.py` | 255-256 | Update `session()` to pass WAL and database |
| `graphiti_core/driver/falkordb_driver.py` | 306-319 | Update `clone()` to pass `wal_writer=self._wal` |
| `graphiti_core/driver/falkordb_driver.py` | 258-265 | Update `close()` to call `await self._wal.close()` |
| `tests/driver/test_wal.py` | N/A (new) | Unit tests for WalWriter + integration tests for WAL with driver |

### Implementation Decisions

#### 1. WAL Close Ownership

**Decision**: Make `WalWriter.close()` idempotent — first call flushes and closes file, subsequent calls are no-ops.

**Rationale**: Multiple driver patterns exist in the codebase:
- **Instance replacement pattern**: When `clone()` replaces `self.driver`, the old driver might call `close()` while the clone still uses the WAL
- **Temporary concurrent clones**: Short-lived clones created by `@handle_multiple_group_ids` decorator never call `close()` — they're discarded after use

Making `close()` idempotent is simpler than reference counting and handles both patterns correctly. The "primary" driver (created with `wal_dir` parameter) will call `close()` on shutdown, and any clone `close()` calls become no-ops.

#### 2. Multi-Database Replay

**Decision**: Support optional database remapping, but default to replaying events to their original `db` field value.

**Rationale**: While not needed for the immediate use case, supporting multi-database replay benefits upstream users. Implementation:
```python
async def replay_wal(
    wal_dir: str,
    host: str,
    port: int,
    database: str | None = None,  # If None, use db from each event
    database_mapping: dict[str, str] | None = None,  # Optional remapping
    from_seq: int = 0,
    dry_run: bool = False
) -> int:
```

If `database` is specified, all events replay to that database (ignore event `db` field). If `database_mapping` is provided, map event `db` values to target databases. Otherwise, use event `db` field as-is.

#### 3. Compression

**Decision**: No compression needed — rely on git's delta compression.

**Rationale**: Git's packfile format already uses delta compression, which is highly effective for the repetitive structure of WAL files. Adding gzip compression would:
- Complicate the implementation (need rotation/compression coordination)
- Reduce git's ability to delta-compress (compressed files don't delta well)
- Add CPU overhead on every write

Git's delta compression is sufficient for the git-friendly backup goal.

</details>


---

<details>
<summary><h2>Implementation Plan</h2></summary>

<!-- kiln:plan -->
# Write-Ahead Log (WAL) for FalkorDB Implementation Plan

## Overview

Implement a write-ahead log (WAL) system that captures all FalkorDB mutations as replayable JSONL files. This enables git-friendly backup of the knowledge graph without storing the 149MB binary database file.

## Current State Analysis

### Key Discoveries:
- **Two mutation paths confirmed**: `FalkorDriver.execute_query()` at line 222-253 and `FalkorDriverSession.run()` at lines 95-106 in `falkordb_driver.py`
- **Datetime serialization handled**: `convert_datetimes_to_strings()` in `utils/datetime_utils.py:45-55` already converts datetimes to ISO 8601
- **`clone()` creates new instance**: Must explicitly pass `wal_writer` parameter (unlike `with_database()` which uses shallow copy)
- **Two driver usage patterns**: Instance replacement (`graphiti.py:886-890`) and temporary concurrent clones (`decorators.py:47-68`)
- **Test patterns established**: `tests/driver/test_falkordb_driver.py` uses `@pytest.mark.asyncio`, `AsyncMock`, `MagicMock`

## Desired End State

After implementation:
1. FalkorDriver accepts optional `wal_dir` parameter to enable WAL logging
2. All mutations (CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE) are captured to JSONL files
3. Index DDL operations are excluded (handled separately by `build_indices_and_constraints()`)
4. WAL files can be replayed to reconstruct the database from scratch
5. `.graphiti/db` can be gitignored; `.graphiti/wal/` becomes the git-tracked source of truth

**Verification**: Run `make test` — all existing tests pass. New WAL tests verify mutation detection, file rotation, and replay functionality.

## What We're NOT Doing

- **No compression** — git's delta compression is sufficient
- **No encryption** — WAL files contain the same data as the database
- **No Neo4j driver changes** — WAL is FalkorDB-specific for now
- **No automatic WAL cleanup/compaction** — files are immutable once written
- **No liminis-framework changes in this PR** — that's a separate downstream task (documented in Step 7)

## Implementation Approach

The WAL is implemented as an optional feature enabled by passing `wal_dir` to `FalkorDriver.__init__()`. The `WalWriter` class handles all file I/O with async locking for concurrency safety. Mutations are logged after successful execution to ensure only committed operations are recorded.

---

### TASK 1: Create WalWriter Class [HIGH PRIORITY]
**Status:** NOT STARTED
**Milestone:** WalWriter class exists with mutation detection, JSONL writing, and file rotation

- [ ] Create `graphiti_core/driver/wal.py` module
- [ ] Implement `WalWriter.__init__(wal_dir, max_events_per_file=10_000)`
- [ ] Implement `is_mutation(cypher)` static method — detect CREATE/MERGE/SET/DELETE/DETACH/DROP/REMOVE
- [ ] Implement `is_index_ddl(cypher)` static method — detect CREATE INDEX, CREATE FULLTEXT INDEX, DROP INDEX, CALL db.idx.fulltext
- [ ] Implement `log_mutation(cypher, params, database)` async method with filtering
- [ ] Implement file rotation when `max_events_per_file` reached
- [ ] Implement sequence number continuity — scan existing files on startup
- [ ] Implement `close()` async method — flush and close current file (idempotent)
- [ ] Add `asyncio.Lock` for thread-safe concurrent writes

**Validation:** Unit tests in `tests/driver/test_wal.py` pass for `is_mutation()`, `is_index_ddl()`, file rotation, and sequence continuity

**Requirements from spec:**
- File naming: `{timestamp}_{session_uuid_short}_{file_seq:04d}.jsonl`
- JSONL format: `{"seq": N, "ts": "ISO8601", "db": "database_name", "cypher": "...", "params": {...}}`
- Mutation keywords: CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE (case-insensitive)
- Index exclusion patterns: `CREATE INDEX`, `CREATE FULLTEXT INDEX`, `DROP INDEX`, `DROP FULLTEXT INDEX`, `CALL db.idx.fulltext.createNodeIndex`

**Files to Create:**
- `graphiti_core/driver/wal.py` — WalWriter class with all mutation detection and file I/O logic

**Implementation Details:**
```python
class WalWriter:
    def __init__(self, wal_dir: str | Path, max_events_per_file: int = 10_000):
        self._wal_dir = Path(wal_dir)
        self._max_events = max_events_per_file
        self._lock = asyncio.Lock()
        self._file: aiofiles.threadpool.AsyncTextIOWrapper | None = None
        self._current_seq = 0
        self._events_in_file = 0
        self._session_id = uuid.uuid4().hex[:6]
        self._file_seq = 0
        self._closed = False
        # Scan for existing seq on init
        
    @staticmethod
    def is_mutation(cypher: str) -> bool:
        # Strip string literals, check for keywords case-insensitively
        
    @staticmethod
    def is_index_ddl(cypher: str) -> bool:
        # Check for index-related patterns
        
    async def log_mutation(self, cypher: str, params: dict, database: str) -> None:
        if not self.is_mutation(cypher) or self.is_index_ddl(cypher):
            return
        # Write JSONL line with seq, ts, db, cypher, params
        
    async def close(self) -> None:
        # Idempotent close — flush and set _closed flag
```

---

### TASK 2: Integrate WAL into FalkorDriver [HIGH PRIORITY]
**Status:** NOT STARTED
**Milestone:** FalkorDriver accepts `wal_dir`/`wal_writer` params and logs mutations from `execute_query()`

- [ ] Add `wal_dir: str | Path | None = None` parameter to `FalkorDriver.__init__()`
- [ ] Add `wal_writer: WalWriter | None = None` parameter to `FalkorDriver.__init__()`
- [ ] Initialize `self._wal` based on parameters (create new or reuse)
- [ ] Import `WalWriter` conditionally to avoid circular imports
- [ ] Instrument `execute_query()` — call `log_mutation()` after successful `graph.query()`
- [ ] Update `close()` — call `await self._wal.close()` if WAL enabled

**Validation:** Existing `test_falkordb_driver.py` tests still pass; manual test shows WAL files created when `wal_dir` provided

**Requirements from spec:**
- `wal_dir` provided → create new `WalWriter(wal_dir)`
- `wal_writer` provided → reuse it (for clones)
- Neither → WAL disabled (`self._wal = None`)
- Log after successful execution only

**Files to Modify:**
- `graphiti_core/driver/falkordb_driver.py` — lines 115-168 (`__init__`), lines 222-236 (`execute_query`), lines 258-265 (`close`)

**Implementation Details:**
```python
# In __init__:
self._wal: WalWriter | None = None
if wal_writer is not None:
    self._wal = wal_writer
elif wal_dir is not None:
    from graphiti_core.driver.wal import WalWriter
    self._wal = WalWriter(wal_dir)

# In execute_query, after line 229:
if self._wal is not None:
    await self._wal.log_mutation(cypher_query_, params, database=self._database)

# In close:
if self._wal is not None:
    await self._wal.close()
```

---

### TASK 3: Integrate WAL into FalkorDriverSession [HIGH PRIORITY]
**Status:** NOT STARTED
**Milestone:** Session-based mutations are logged to WAL

- [ ] Add `wal: WalWriter | None = None` parameter to `FalkorDriverSession.__init__()`
- [ ] Add `database: str | None = None` parameter to `FalkorDriverSession.__init__()`
- [ ] Store as `self._wal` and `self._database` instance variables
- [ ] Update `FalkorDriver.session()` to pass `wal=self._wal, database=database or self._database`
- [ ] Instrument `run()` — call `log_mutation()` after each successful `graph.query()` in both branches

**Validation:** Session-based operations (list queries) are logged to WAL

**Requirements from spec:**
- Session must receive WAL and database from parent driver
- Both single-query and list-query branches in `run()` must log mutations

**Files to Modify:**
- `graphiti_core/driver/falkordb_driver.py` — lines 74-77 (`FalkorDriverSession.__init__`), lines 95-106 (`run`), lines 255-256 (`session`)

**Implementation Details:**
```python
# FalkorDriverSession.__init__:
def __init__(self, graph: FalkorGraph, wal: WalWriter | None = None, database: str | None = None):
    self.graph = graph
    self._wal = wal
    self._database = database

# FalkorDriverSession.run, after each graph.query():
if self._wal is not None:
    await self._wal.log_mutation(str(cypher), params, database=self._database or '')

# FalkorDriver.session:
def session(self, database: str | None = None) -> GraphDriverSession:
    return FalkorDriverSession(
        self._get_graph(database),
        wal=self._wal,
        database=database or self._database
    )
```

---

### TASK 4: Update clone() to Share WAL [HIGH PRIORITY]
**Status:** NOT STARTED
**Milestone:** Cloned drivers share the same WAL instance

- [ ] Update `clone()` method to pass `wal_writer=self._wal` when creating new FalkorDriver instances
- [ ] Handle all three branches: same database, default_group_id, and custom database

**Validation:** Cloned driver mutations appear in the same WAL files as parent driver

**Requirements from spec:**
- All cloned drivers write to the same WAL with their respective `database` field
- `with_database()` already works via shallow copy — no changes needed

**Files to Modify:**
- `graphiti_core/driver/falkordb_driver.py` — lines 306-319 (`clone`)

**Implementation Details:**
```python
def clone(self, database: str) -> 'GraphDriver':
    if database == self._database:
        cloned = self
    elif database == self.default_group_id:
        cloned = FalkorDriver(falkor_db=self.client, wal_writer=self._wal)
    else:
        cloned = FalkorDriver(falkor_db=self.client, database=database, wal_writer=self._wal)
    return cloned
```

---

### TASK 5: Create WAL Replay Script [MEDIUM PRIORITY]
**Status:** NOT STARTED
**Milestone:** CLI script can replay WAL files to reconstruct database

- [ ] Create `graphiti_core/driver/wal_replay.py` module
- [ ] Implement `replay_wal(wal_dir, host, port, database, from_seq, dry_run)` async function
- [ ] Read all `.jsonl` files sorted by filename
- [ ] Parse each line and execute cypher against appropriate database
- [ ] Support `from_seq` for partial replay / resume
- [ ] Support `dry_run` mode that prints queries without executing
- [ ] Support optional `database_mapping` for remapping database names
- [ ] Call `build_indices_and_constraints()` before replay
- [ ] Add CLI entry point with argparse

**Validation:** Replay empty DB with WAL files → query results match original DB

**Requirements from spec:**
- Connect to fresh FalkorDB with WAL disabled
- Build indices first
- Execute events in sequence order
- Skip events with `seq < from_seq`

**Files to Create:**
- `graphiti_core/driver/wal_replay.py` — replay function + CLI

**Implementation Details:**
```python
async def replay_wal(
    wal_dir: str | Path,
    host: str = 'localhost',
    port: int = 6379,
    database: str | None = None,
    database_mapping: dict[str, str] | None = None,
    from_seq: int = 0,
    dry_run: bool = False,
) -> int:
    """Replay WAL files to reconstruct database. Returns count of events replayed."""
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay WAL files to FalkorDB')
    parser.add_argument('wal_dir', help='Directory containing WAL files')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--database', help='Target database (default: use db from events)')
    parser.add_argument('--from-seq', type=int, default=0, help='Start from sequence number')
    parser.add_argument('--dry-run', action='store_true', help='Print queries without executing')
    # ...
```

---

### TASK 6: Write Unit Tests [MEDIUM PRIORITY]
**Status:** NOT STARTED
**Milestone:** Comprehensive test coverage for WAL functionality

- [ ] Create `tests/driver/test_wal.py`
- [ ] Test `is_mutation()` with various query types (MATCH, CREATE, MERGE, SET, DELETE, etc.)
- [ ] Test `is_index_ddl()` with all index patterns from `graph_queries.py`
- [ ] Test file rotation after `max_events_per_file`
- [ ] Test sequence continuity across file boundaries
- [ ] Test sequence continuity on restart (scan existing files)
- [ ] Test concurrent writes don't interleave
- [ ] Test `close()` idempotency
- [ ] Test WAL integration in FalkorDriver (mock-based)
- [ ] Test WAL integration in FalkorDriverSession (mock-based)
- [ ] Test clone() shares WAL instance

**Validation:** `make test` passes, new tests provide >90% coverage of WAL code

**Requirements from spec:**
- Follow existing patterns from `test_falkordb_driver.py`
- Use `@pytest.mark.asyncio`, `AsyncMock`, `MagicMock`
- Use temp directories for file-based tests

**Files to Create:**
- `tests/driver/test_wal.py` — comprehensive unit tests

**Implementation Details:**
Test categories:
1. `TestIsMutation` — query classification
2. `TestIsIndexDDL` — index DDL detection
3. `TestWalWriter` — file operations, rotation, sequence
4. `TestWalDriverIntegration` — mock-based driver integration
5. `TestWalReplay` — replay functionality

---

### TASK 7: Integration Testing [LOW PRIORITY]
**Status:** NOT STARTED
**Milestone:** End-to-end WAL capture and replay verified

- [ ] Create integration test that requires FalkorDB connection
- [ ] Start driver with WAL enabled
- [ ] Execute various mutations (create entities, edges, delete)
- [ ] Verify WAL files contain expected mutations
- [ ] Replay to fresh database
- [ ] Compare query results between original and replayed DB

**Validation:** Integration test passes with real FalkorDB instance

**Requirements from spec:**
- Mark with `@unittest.skipIf(not HAS_FALKORDB, ...)`
- Use `pytest.importorskip('falkordb')`
- Use temp directory for WAL files

**Files to Modify:**
- `tests/driver/test_wal.py` — add integration test class

---

## Testing Strategy

### Unit Tests:
- `is_mutation()` correctly classifies: MATCH (false), CREATE (true), MERGE (true), SET (true), DELETE (true), DETACH DELETE (true), DROP (true), REMOVE (true)
- `is_index_ddl()` detects: CREATE INDEX FOR, CREATE FULLTEXT INDEX, DROP INDEX ON, DROP FULLTEXT INDEX, CALL db.idx.fulltext.createNodeIndex
- File rotation creates new file after `max_events_per_file`
- Sequence numbers are monotonic across files
- Sequence numbers survive restart by scanning existing files
- `close()` is idempotent — multiple calls don't error

### Integration Tests:
- WAL files created when `wal_dir` passed to FalkorDriver
- Mutations logged, reads not logged
- Index DDL not logged
- Replay reconstructs semantically equivalent database

### Manual Testing Steps:
1. Start FalkorDB locally: `docker run -p 6379:6379 falkordb/falkordb`
2. Create driver with `wal_dir='/tmp/test_wal'`
3. Execute `driver.execute_query("CREATE (n:Test {name: 'test'})")`
4. Verify `/tmp/test_wal/*.jsonl` contains the mutation
5. Drop database, replay WAL
6. Query for test node — should exist

## Performance Considerations

- **Write overhead**: Minimal — async file writes with buffering
- **WAL file size**: ~15-20KB per entity mutation (with 1024-dim embeddings), ~0.5-1KB for other mutations
- **Concurrency**: `asyncio.Lock` prevents interleaved writes but serializes logging (acceptable given I/O bottleneck)
- **Git efficiency**: Small, append-only files delta-compress well in packfiles

## Migration Notes

- No migration needed — WAL is opt-in via `wal_dir` parameter
- Existing deployments without WAL continue to work unchanged
- To enable: pass `wal_dir` when constructing `FalkorDriver`
- Downstream task (liminis-framework): Update `graphiti_service.py` line ~462 to pass `wal_dir`
<!-- /kiln:plan -->

</details>