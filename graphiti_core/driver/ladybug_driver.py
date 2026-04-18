"""
LadybugDB driver for graphiti-core.

Drop-in replacement using LadybugDB (real-ladybug on PyPI),
the community fork of KuzuDB.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import real_ladybug as kuzu

if TYPE_CHECKING:
    from graphiti_core.driver.wal import WalWriter

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.ladybug.operations.community_edge_ops import (
    LadybugCommunityEdgeOperations,
)
from graphiti_core.driver.ladybug.operations.community_node_ops import (
    LadybugCommunityNodeOperations,
)
from graphiti_core.driver.ladybug.operations.entity_edge_ops import LadybugEntityEdgeOperations
from graphiti_core.driver.ladybug.operations.entity_node_ops import LadybugEntityNodeOperations
from graphiti_core.driver.ladybug.operations.episode_node_ops import LadybugEpisodeNodeOperations
from graphiti_core.driver.ladybug.operations.episodic_edge_ops import LadybugEpisodicEdgeOperations
from graphiti_core.driver.ladybug.operations.graph_ops import LadybugGraphMaintenanceOperations
from graphiti_core.driver.ladybug.operations.has_episode_edge_ops import (
    LadybugHasEpisodeEdgeOperations,
)
from graphiti_core.driver.ladybug.operations.next_episode_edge_ops import (
    LadybugNextEpisodeEdgeOperations,
)
from graphiti_core.driver.ladybug.operations.saga_node_ops import LadybugSagaNodeOperations
from graphiti_core.driver.ladybug.operations.search_ops import LadybugSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.wal_replay_helpers import expand_bulk_property_set, strip_vecf32_wrappers
from graphiti_core.embedder.client import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# LadybugDB schema DDL.
SCHEMA_QUERIES = f"""
    CREATE NODE TABLE IF NOT EXISTS Episodic (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        source STRING,
        source_description STRING,
        content STRING,
        content_embedding FLOAT[{EMBEDDING_DIM}],
        valid_at TIMESTAMP,
        entity_edges STRING[]
    );
    CREATE NODE TABLE IF NOT EXISTS Entity (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        labels STRING[],
        created_at TIMESTAMP,
        name_embedding FLOAT[{EMBEDDING_DIM}],
        summary STRING,
        attributes STRING
    );
    CREATE NODE TABLE IF NOT EXISTS Community (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        name_embedding FLOAT[{EMBEDDING_DIM}],
        summary STRING
    );
    CREATE NODE TABLE IF NOT EXISTS RelatesToNode_ (
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP,
        name STRING,
        fact STRING,
        fact_embedding FLOAT[{EMBEDDING_DIM}],
        episodes STRING[],
        expired_at TIMESTAMP,
        valid_at TIMESTAMP,
        invalid_at TIMESTAMP,
        attributes STRING
    );
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO RelatesToNode_,
        FROM RelatesToNode_ TO Entity
    );
    CREATE REL TABLE IF NOT EXISTS MENTIONS(
        FROM Episodic TO Entity,
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_MEMBER(
        FROM Community TO Entity,
        FROM Community TO Community,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE NODE TABLE IF NOT EXISTS Saga (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_EPISODE(
        FROM Saga TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS NEXT_EPISODE(
        FROM Episodic TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
"""


def _fix_record_timestamps(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise naive datetime values to UTC.

    LadybugDB (like KuzuDB) returns TIMESTAMP columns as naive datetimes.
    Graphiti compares these against timezone-aware datetimes elsewhere
    (e.g. edge_operations.py), which causes a crash. This helper attaches
    UTC tzinfo to any naive datetime found in a result record.
    """
    for key, value in record.items():
        if isinstance(value, datetime) and value.tzinfo is None:
            record[key] = value.replace(tzinfo=timezone.utc)
        elif isinstance(value, dict):
            _fix_record_timestamps(value)
    return record


class LadybugDriver(GraphDriver):
    # Provider identity for LadybugDB
    provider: GraphProvider = GraphProvider.LADYBUG
    aoss_client: None = None

    def __init__(
        self,
        db: str = ':memory:',
        max_concurrent_queries: int = 1,
        wal_dir: str | Path | None = None,
        wal_writer: WalWriter | None = None,
    ):
        super().__init__()
        self._database = ''  # LadybugDB is single-database; needed by graphiti.py
        self.db = kuzu.Database(db)

        self.setup_schema()

        self.client = kuzu.AsyncConnection(self.db, max_concurrent_queries=max_concurrent_queries)

        # Initialize WAL writer (same pattern as FalkorDriver).
        self._wal: WalWriter | None = None
        self._wal_owner = False
        if wal_writer is not None:
            self._wal = wal_writer
        elif wal_dir is not None:
            from graphiti_core.driver.wal import WalWriter

            self._wal = WalWriter(wal_dir)
            self._wal_owner = True

        # Ladybug operations — they only speak Cypher.
        self._entity_node_ops = LadybugEntityNodeOperations()
        self._episode_node_ops = LadybugEpisodeNodeOperations()
        self._community_node_ops = LadybugCommunityNodeOperations()
        self._saga_node_ops = LadybugSagaNodeOperations()
        self._entity_edge_ops = LadybugEntityEdgeOperations()
        self._episodic_edge_ops = LadybugEpisodicEdgeOperations()
        self._community_edge_ops = LadybugCommunityEdgeOperations()
        self._has_episode_edge_ops = LadybugHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = LadybugNextEpisodeEdgeOperations()
        self._search_ops = LadybugSearchOperations()
        self._graph_ops = LadybugGraphMaintenanceOperations()

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]] | list[list[dict[str, Any]]], None, None]:
        params = dict(kwargs)
        # LadybugDB (like Kuzu) does not support these Neo4j-specific parameters.
        params.pop('database_', None)
        params.pop('routing_', None)

        if logger.isEnabledFor(logging.DEBUG):
            is_write = any(
                kw in cypher_query_.upper() for kw in ('MERGE', 'CREATE', 'SET', 'DELETE')
            )
            if is_write:
                logger.debug('LadybugDB WRITE query executed')

        try:
            results = await self.client.execute(cypher_query_, parameters=params)
        except Exception as e:
            params = {k: (v[:5] if isinstance(v, list) else v) for k, v in params.items()}
            logger.error(f'Error executing LadybugDB query: {e}\n{cypher_query_}\n{params}')
            raise

        # Log mutation to WAL after successful execution, before checking results.
        # This ensures DELETE/DETACH DELETE queries (which return no rows) are logged.
        if self._wal is not None:
            await self._wal.log_mutation(cypher_query_, cast(dict[str, Any], params), database='')

        if not results:
            return [], None, None

        if isinstance(results, list):
            dict_results = [
                [_fix_record_timestamps(row) for row in result.rows_as_dict()] for result in results
            ]
        else:
            dict_results = [_fix_record_timestamps(row) for row in results.rows_as_dict()]
        return dict_results, None, None  # type: ignore

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return LadybugDriverSession(self)

    async def close(self):
        if self._wal is not None and self._wal_owner:
            await self._wal.close()
        # Explicitly close the AsyncConnection before the Database so that all
        # in-flight queries are drained and Kuzu's WAL is checkpointed into the
        # main database file before the process exits.  Relying on GC is unsafe
        # under SIGTERM → SIGKILL escalations with short timeouts (e.g. 5 s
        # systemd/container shutdown budgets) because GC may never run, leaving
        # db.wal uncheckpointed and the database unreadable on next startup.
        # Both close() calls are synchronous and idempotent.
        # Use try/finally so db.close() (WAL checkpoint) runs even if
        # client.close() raises an unexpected exception.
        try:
            self.client.close()
        finally:
            self.db.close()

    async def rotate_wal(self) -> None:
        """Rotate the WAL file, closing the current tip file."""
        if self._wal is not None:
            await self._wal.rotate()

    def wal_chunk(self):
        """Return a WAL chunk context manager for batching mutations per episode.

        Returns WalWriter.chunk() when WAL is enabled, otherwise a no-op context.
        All mutations executed inside the context are buffered and flushed to a
        single JSONL file on clean exit; discarded on exception.
        """
        if self._wal is not None:
            return self._wal.chunk()
        return nullcontext()

    async def delete_all_indexes(self) -> None:
        """No-op for LadybugDB; required to satisfy GraphDriver interface."""
        pass

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Load FTS extension on the async connection — setup_schema() loaded it
        # on the sync connection, but extensions are per-connection in LadybugDB.
        try:
            await self.client.execute('LOAD EXTENSION FTS;')
            logger.info('FTS extension loaded on async connection')
        except Exception as e:
            logger.warning(f'Could not load FTS extension on async connection: {e}')

        # Load vector extension on async connection (same dual-load pattern as FTS).
        try:
            await self.client.execute('LOAD EXTENSION vector;')
            logger.info('vector extension loaded on async connection')
        except Exception as e:
            logger.warning(f'Could not load vector extension on async connection: {e}')

        # Create FTS indexes
        # but graphiti's dedup pipeline needs fulltext search to work.
        from graphiti_core.graph_queries import get_fulltext_indices, get_vector_indices

        for query in get_fulltext_indices(GraphProvider.LADYBUG):
            try:
                await self.client.execute(query)
                logger.info(f'Created FTS index: {query[:80]}')
            except Exception as e:
                if 'already exists' in str(e).lower():
                    logger.debug(f'FTS index already exists: {query[:80]}')
                else:
                    logger.error(f'Failed to create FTS index: {e}\n{query}')

        # Create HNSW vector indexes for similarity search.
        for query in get_vector_indices(GraphProvider.LADYBUG):
            try:
                await self.client.execute(query)
                logger.info(f'Created vector index: {query[:80]}')
            except Exception as e:
                if 'already exists' in str(e).lower():
                    logger.debug(f'Vector index already exists: {query[:80]}')
                else:
                    logger.error(f'Failed to create vector index: {e}\n{query}')

    def setup_schema(self):
        conn = kuzu.Connection(self.db)
        # Load FTS extension before creating schema — required for
        # fulltext index creation in build_indices_and_constraints().
        try:
            conn.execute('INSTALL FTS; LOAD EXTENSION FTS;')
        except Exception as e:
            # FTS may already be installed/loaded
            logger.debug(f'FTS extension setup: {e}')
            try:
                conn.execute('LOAD EXTENSION FTS;')
            except Exception as e_load:
                logger.warning(
                    f'Could not load FTS extension — fulltext search will be unavailable: {e_load}'
                )
        # Load vector extension for HNSW index support (same pattern as FTS above).
        try:
            conn.execute('INSTALL vector; LOAD EXTENSION vector;')
            logger.info('vector extension loaded on sync connection')
        except Exception as e:
            logger.debug(f'vector extension setup: {e}')
            try:
                conn.execute('LOAD EXTENSION vector;')
                logger.info('vector extension loaded on sync connection')
            except Exception as e_load:
                logger.warning(
                    f'Could not load vector extension — HNSW indexes will be unavailable: {e_load}'
                )
        conn.execute(SCHEMA_QUERIES)
        conn.close()


class LadybugDriverSession(GraphDriverSession):
    provider = GraphProvider.LADYBUG

    def __init__(self, driver: LadybugDriver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def close(self):
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # WAL logging is handled by driver.execute_query() — no duplicate logging here.
        if isinstance(query, list):
            for cypher, params in query:
                await self.driver.execute_query(cypher, **params)
        else:
            await self.driver.execute_query(query, **kwargs)
        return None


import re

# ISO-8601 pattern for detecting timestamp strings in WAL params
_ISO_TS_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'
)

# Known timestamp field names in the graphiti schema
_TIMESTAMP_FIELDS = frozenset({
    'created_at', 'valid_at', 'invalid_at', 'expired_at',
})


def _is_timestamp_field(key: str) -> bool:
    """Check if a param key refers to a timestamp field.

    Handles both flat keys ('created_at') and prefixed keys from
    expand_bulk_property_set ('props_created_at').
    """
    if key in _TIMESTAMP_FIELDS:
        return True
    # After bulk SET expansion, keys are prefixed: props_created_at
    suffix = key.split('_', 1)[1] if '_' in key else ''
    return suffix in _TIMESTAMP_FIELDS


def _deserialize_wal_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert WAL JSON params back to Python types for LadybugDB.

    WAL serializes datetime objects as ISO-8601 strings. LadybugDB
    requires native datetime objects for TIMESTAMP columns. This function
    detects timestamp fields and converts them back.

    Handles both flat param names ('created_at') and prefixed names
    from bulk SET expansion ('props_created_at').
    """
    converted = {}
    for key, value in params.items():
        if (
            isinstance(value, str)
            and _is_timestamp_field(key)
            and _ISO_TS_RE.match(value)
        ):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                converted[key] = dt
                continue
            except (ValueError, TypeError):
                pass  # Fall through to keep as string
        converted[key] = value
    return converted


async def replay_wal_ladybug(
    wal_dir: str | Path,
    db: str,
    from_seq: int = 0,
    dry_run: bool = False,
    batch_size: int = 10000,
) -> int:
    """Replay WAL files into a LadybugDB database.

    Creates a LadybugDriver WITHOUT WAL (to avoid re-logging replayed
    mutations), reads JSONL files in filename order, and executes each
    mutation.

    Uses BEGIN TRANSACTION / COMMIT batching for ~60x speedup over
    per-mutation implicit transactions. On batch failure, uses binary-split
    retry to isolate the failing mutation(s) while preserving all good
    mutations. See LadybugDB/ladybug#386.

    FalkorDB-era WAL entries (produced by wal_dump.py) wrap embedding
    parameters in `vecf32($...)` which LadybugDB does not support; those
    wrappers are stripped in-place before execution. Pure LadybugDB-era
    entries are unaffected.

    Args:
        wal_dir: Directory containing WAL .jsonl files.
        db: LadybugDB database path.
        from_seq: Skip events with seq < from_seq (for partial replay).
        dry_run: If True, parse and validate but don't execute.
        batch_size: Mutations per transaction commit (default 10000).

    Returns:
        Number of mutations replayed.
    """
    import real_ladybug as ladybug

    wal_path = Path(wal_dir)
    if not wal_path.is_dir():
        raise FileNotFoundError(f'WAL directory not found: {wal_path}')

    wal_files = sorted(wal_path.glob('*.jsonl'))
    if not wal_files:
        logger.info('No WAL files found in %s', wal_path)
        return 0

    driver: LadybugDriver | None = None
    conn = None
    replayed = 0
    skipped = 0
    errors = 0

    def _replay_batch(
        conn: ladybug.Connection,
        mutations: list[tuple[int, str, dict]],
    ) -> tuple[int, int]:
        """Replay a batch in a single transaction with binary-split retry."""
        if not mutations:
            return 0, 0
        try:
            conn.execute('BEGIN TRANSACTION')
            for _seq, cypher, params in mutations:
                conn.execute(cypher, parameters=params)
            conn.execute('COMMIT')
            return len(mutations), 0
        except Exception as e:
            try:
                conn.execute('ROLLBACK')
            except Exception:
                pass
            if len(mutations) == 1:
                logger.warning('Replay skip at seq=%d: %s', mutations[0][0], e)
                return 0, 1
            mid = len(mutations) // 2
            left_r, left_e = _replay_batch(conn, mutations[:mid])
            right_r, right_e = _replay_batch(conn, mutations[mid:])
            return left_r + right_r, left_e + right_e

    try:
        if not dry_run:
            # No wal_dir — we don't want to re-log replayed mutations
            driver = LadybugDriver(db=db)
            conn = ladybug.Connection(driver.db)
            # NOTE: build_indices_and_constraints is called AFTER replay,
            # not before. LadybugDB HNSW indexes block in-place vector column
            # updates via MERGE...SET, so we bulk-load data first, then
            # create indexes on the final state.

        batch: list[tuple[int, str, dict]] = []

        for wal_file in wal_files:
            logger.info('Replaying %s...', wal_file.name)

            with open(wal_file, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error('Parse error in %s:%d: %s', wal_file.name, line_num, e)
                        errors += 1
                        continue

                    seq = entry.get('seq', -1)
                    if seq < from_seq:
                        skipped += 1
                        continue

                    cypher = strip_vecf32_wrappers(entry['cypher'])
                    params = entry.get('params', {})
                    cypher, params = expand_bulk_property_set(cypher, params)
                    params = _deserialize_wal_params(params)

                    if dry_run:
                        logger.debug('DRY RUN seq=%d: %s', seq, cypher[:80])
                        replayed += 1
                        continue

                    if conn is None:
                        raise RuntimeError('Connection is not initialized for WAL replay')

                    batch.append((seq, cypher, params))

                    if len(batch) >= batch_size:
                        batch_r, batch_e = _replay_batch(conn, batch)
                        replayed += batch_r
                        errors += batch_e
                        batch = []
                        logger.info(
                            'Committed batch: %d total replayed (%d errors)',
                            replayed, errors,
                        )

        # Flush remaining batch
        if not dry_run and conn is not None and batch:
            batch_r, batch_e = _replay_batch(conn, batch)
            replayed += batch_r
            errors += batch_e

    finally:
        if conn is not None:
            conn.close()
        if driver is not None:
            if not dry_run and replayed > 0:
                logger.info('Building indices and constraints on replayed data...')
                await driver.build_indices_and_constraints()
            await driver.close()

    logger.info(
        'Replay complete: %d replayed, %d skipped (seq < %d), %d errors',
        replayed,
        skipped,
        from_seq,
        errors,
    )
    return replayed
