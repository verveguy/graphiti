"""
LadybugDB driver for graphiti-core.

Drop-in replacement for KuzuDriver using LadybugDB (real-ladybug on PyPI),
the community fork of KuzuDB. Structurally identical to kuzu_driver.py with:
  1. `import real_ladybug as kuzu`
  2. Timezone fix in execute_query() for issue #893/#920
  3. provider kept as KUZU so all KUZU-specific branches continue to fire
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import real_ladybug as kuzu

if TYPE_CHECKING:
    from graphiti_core.driver.wal import WalWriter

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.driver.kuzu.operations.community_edge_ops import KuzuCommunityEdgeOperations
from graphiti_core.driver.kuzu.operations.community_node_ops import KuzuCommunityNodeOperations
from graphiti_core.driver.kuzu.operations.entity_edge_ops import KuzuEntityEdgeOperations
from graphiti_core.driver.kuzu.operations.entity_node_ops import KuzuEntityNodeOperations
from graphiti_core.driver.kuzu.operations.episode_node_ops import KuzuEpisodeNodeOperations
from graphiti_core.driver.kuzu.operations.episodic_edge_ops import KuzuEpisodicEdgeOperations
from graphiti_core.driver.kuzu.operations.graph_ops import KuzuGraphMaintenanceOperations
from graphiti_core.driver.kuzu.operations.has_episode_edge_ops import KuzuHasEpisodeEdgeOperations
from graphiti_core.driver.kuzu.operations.next_episode_edge_ops import (
    KuzuNextEpisodeEdgeOperations,
)
from graphiti_core.driver.kuzu.operations.saga_node_ops import KuzuSagaNodeOperations
from graphiti_core.driver.kuzu.operations.search_ops import KuzuSearchOperations
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

logger = logging.getLogger(__name__)

# Schema is identical to kuzu_driver.py — LadybugDB uses the same Cypher DDL.
SCHEMA_QUERIES = f"""
    CREATE NODE TABLE IF NOT EXISTS Episodic (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        source STRING,
        source_description STRING,
        content STRING,
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
    # Keep provider as KUZU so all KUZU-specific branches in graphiti
    # (e.g. RelatesToNode_ handling, FTS index loading) continue to fire.
    provider: GraphProvider = GraphProvider.KUZU
    aoss_client: None = None

    def __init__(
        self,
        db: str = ':memory:',
        max_concurrent_queries: int = 1,
        wal_dir: str | Path | None = None,
        wal_writer: WalWriter | None = None,
    ):
        super().__init__()
        self._database = ''  # Kuzu/LadybugDB is single-database; needed by graphiti.py
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

        # Reuse all Kuzu operations — they only speak Cypher, no kuzu import.
        self._entity_node_ops = KuzuEntityNodeOperations()
        self._episode_node_ops = KuzuEpisodeNodeOperations()
        self._community_node_ops = KuzuCommunityNodeOperations()
        self._saga_node_ops = KuzuSagaNodeOperations()
        self._entity_edge_ops = KuzuEntityEdgeOperations()
        self._episodic_edge_ops = KuzuEpisodicEdgeOperations()
        self._community_edge_ops = KuzuCommunityEdgeOperations()
        self._has_episode_edge_ops = KuzuHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = KuzuNextEpisodeEdgeOperations()
        self._search_ops = KuzuSearchOperations()
        self._graph_ops = KuzuGraphMaintenanceOperations()

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
            is_write = any(kw in cypher_query_.upper() for kw in ('MERGE', 'CREATE', 'SET', 'DELETE'))
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
            await self._wal.log_mutation(
                cypher_query_, cast(dict[str, Any], params), database=''
            )

        if not results:
            return [], None, None

        if isinstance(results, list):
            dict_results = [
                [_fix_record_timestamps(row) for row in result.rows_as_dict()]
                for result in results
            ]
        else:
            dict_results = [_fix_record_timestamps(row) for row in results.rows_as_dict()]
        return dict_results, None, None  # type: ignore

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return LadybugDriverSession(self)

    async def close(self):
        if self._wal is not None and self._wal_owner:
            await self._wal.close()

    async def rotate_wal(self) -> None:
        """Rotate the WAL file, closing the current tip file."""
        if self._wal is not None:
            await self._wal.rotate()

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

        # Create FTS indexes — the original KuzuDriver was a no-op here,
        # but graphiti's dedup pipeline needs fulltext search to work.
        from graphiti_core.graph_queries import get_fulltext_indices

        for query in get_fulltext_indices(GraphProvider.KUZU):
            try:
                await self.client.execute(query)
                logger.info(f'Created FTS index: {query[:80]}')
            except Exception as e:
                if 'already exists' in str(e).lower():
                    logger.debug(f'FTS index already exists: {query[:80]}')
                else:
                    logger.error(f'Failed to create FTS index: {e}\n{query}')

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
                logger.warning(f'Could not load FTS extension — fulltext search will be unavailable: {e_load}')
        conn.execute(SCHEMA_QUERIES)
        conn.close()


class LadybugDriverSession(GraphDriverSession):
    provider = GraphProvider.KUZU

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


async def replay_wal_ladybug(
    wal_dir: str | Path,
    db: str,
    from_seq: int = 0,
    dry_run: bool = False,
) -> int:
    """Replay WAL files into a LadybugDB database.

    Creates a LadybugDriver WITHOUT WAL (to avoid re-logging replayed
    mutations), reads JSONL files in filename order, and executes each
    mutation.

    Args:
        wal_dir: Directory containing WAL .jsonl files.
        db: LadybugDB database path.
        from_seq: Skip events with seq < from_seq (for partial replay).
        dry_run: If True, parse and validate but don't execute.

    Returns:
        Number of mutations replayed.
    """
    wal_path = Path(wal_dir)
    if not wal_path.is_dir():
        raise FileNotFoundError(f'WAL directory not found: {wal_path}')

    wal_files = sorted(wal_path.glob('*.jsonl'))
    if not wal_files:
        logger.info('No WAL files found in %s', wal_path)
        return 0

    driver: LadybugDriver | None = None
    replayed = 0
    skipped = 0
    errors = 0

    try:
        if not dry_run:
            # No wal_dir — we don't want to re-log replayed mutations
            driver = LadybugDriver(db=db)
            await driver.build_indices_and_constraints()

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

                    cypher = entry['cypher']
                    params = entry.get('params', {})

                    if dry_run:
                        logger.debug('DRY RUN seq=%d: %s', seq, cypher[:80])
                        replayed += 1
                        continue

                    try:
                        if driver is None:
                            raise RuntimeError('LadybugDriver is not initialized for WAL replay')
                        await driver.execute_query(cypher, **params)
                        replayed += 1
                    except Exception as e:
                        logger.error('Replay error at seq=%d: %s\n  %s', seq, e, cypher[:200])
                        errors += 1

    finally:
        if driver is not None:
            await driver.close()

    logger.info(
        'Replay complete: %d replayed, %d skipped (seq < %d), %d errors',
        replayed, skipped, from_seq, errors,
    )
    return replayed
