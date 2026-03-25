"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from graphiti_core.driver.wal import DEFAULT_MAX_EVENTS_PER_FILE, WalWriter

logger = logging.getLogger(__name__)

# Properties that are stored as vector embeddings in FalkorDB.
# WAL entries emitted by this module already wrap these in vecf32($...)
# in the generated Cypher; wal_replay.py executes that Cypher as-is.
_VECTOR_PROPERTIES = frozenset({'name_embedding', 'fact_embedding'})

# Internal FalkorDB properties to exclude from dump
_INTERNAL_PROPERTIES = frozenset({'__name_embedding_dim', '__fact_embedding_dim'})


async def dump_wal(
    host: str = 'localhost',
    port: int = 6379,
    database: str = 'default_db',
    wal_dir: str | Path = '.',
    max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE,
) -> int:
    """
    Dump the current state of a FalkorDB database as WAL entries.

    Reads all nodes and relationships from the database and writes
    them as MERGE/SET Cypher mutations to WAL files. The resulting
    WAL can be replayed to reconstruct an equivalent database.

    Nodes are dumped before relationships to ensure MATCH clauses
    in edge mutations find their endpoints.

    Args:
        host: FalkorDB host.
        port: FalkorDB port.
        database: Database name to dump.
        wal_dir: Directory to write WAL files to.
        max_events_per_file: Max events per WAL file.

    Returns:
        Number of mutations written.
    """
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = FalkorDriver(host=host, port=int(port), database=database)
    wal = WalWriter(wal_dir, max_events_per_file=max_events_per_file)

    count = 0

    try:
        # Phase 1: Dump all nodes
        count += await _dump_nodes(driver, wal, database)

        # Phase 2: Dump all relationships
        count += await _dump_relationships(driver, wal, database)

    finally:
        await wal.close()
        await driver.close()

    logger.info('Dump complete: %d mutations written to %s', count, wal_dir)
    return count


async def _dump_nodes(driver: Any, wal: WalWriter, database: str) -> int:
    """Dump all nodes as MERGE/SET WAL entries."""
    result = await driver.execute_query(
        'MATCH (n) RETURN id(n) AS _id, labels(n) AS _labels, properties(n) AS _props'
    )
    if not result:
        return 0

    records, _, _ = result
    count = 0

    for record in records:
        labels = record['_labels']
        props = record['_props']

        # Filter internal properties
        props = {k: v for k, v in props.items() if k not in _INTERNAL_PROPERTIES}

        if not props.get('uuid'):
            logger.warning('Skipping node without uuid: labels=%s', labels)
            continue

        # Build label string for Cypher (e.g., ":Entity:Person")
        label_str = ':'.join(labels) if labels else 'Node'

        # Separate vector properties for vecf32() wrapping
        vector_props = {}
        scalar_props = {}
        for k, v in props.items():
            if k in _VECTOR_PROPERTIES and v is not None:
                vector_props[k] = v
            else:
                scalar_props[k] = v

        # Build Cypher: MERGE on uuid, SET all properties
        cypher = f'MERGE (n:{label_str} {{uuid: $uuid}}) SET n = $props'
        params: dict[str, Any] = {'uuid': props['uuid'], 'props': scalar_props}

        # Add vector properties with vecf32()
        for vk, vv in vector_props.items():
            cypher += f' SET n.{vk} = vecf32(${vk})'
            params[vk] = vv

        await wal.log_mutation(cypher, params, database=database)
        count += 1

    logger.info('Dumped %d nodes', count)
    return count


async def _dump_relationships(driver: Any, wal: WalWriter, database: str) -> int:
    """Dump all relationships as MERGE/SET WAL entries."""
    result = await driver.execute_query("""
        MATCH (src)-[r]->(dst)
        RETURN
            type(r) AS _type,
            src.uuid AS _src_uuid,
            dst.uuid AS _dst_uuid,
            properties(r) AS _props,
            labels(src) AS _src_labels,
            labels(dst) AS _dst_labels
    """)
    if not result:
        return 0

    records, _, _ = result
    count = 0

    for record in records:
        rel_type = record['_type']
        src_uuid = record['_src_uuid']
        dst_uuid = record['_dst_uuid']
        props = record['_props']
        src_labels = record['_src_labels']
        dst_labels = record['_dst_labels']

        # Filter internal properties
        props = {k: v for k, v in props.items() if k not in _INTERNAL_PROPERTIES}

        if not src_uuid or not dst_uuid:
            logger.warning('Skipping relationship without endpoint uuids: type=%s', rel_type)
            continue

        # Use first label for MATCH (sufficient for uuid lookup)
        src_label = src_labels[0] if src_labels else 'Node'
        dst_label = dst_labels[0] if dst_labels else 'Node'

        # Separate vector properties
        vector_props = {}
        scalar_props = {}
        for k, v in props.items():
            if k in _VECTOR_PROPERTIES and v is not None:
                vector_props[k] = v
            else:
                scalar_props[k] = v

        # Build MERGE key — use uuid if available, otherwise match on endpoints
        if props.get('uuid'):
            merge_key = '{uuid: $uuid}'
            params: dict[str, Any] = {
                'src_uuid': src_uuid,
                'dst_uuid': dst_uuid,
                'uuid': props['uuid'],
                'props': scalar_props,
            }
        else:
            merge_key = ''
            params = {
                'src_uuid': src_uuid,
                'dst_uuid': dst_uuid,
                'props': scalar_props,
            }

        cypher = (
            f'MATCH (src:{src_label} {{uuid: $src_uuid}}) '
            f'MATCH (dst:{dst_label} {{uuid: $dst_uuid}}) '
            f'MERGE (src)-[r:{rel_type} {merge_key}]->(dst) '
            f'SET r = $props'
        )

        # Add vector properties with vecf32()
        for vk, vv in vector_props.items():
            cypher += f' SET r.{vk} = vecf32(${vk})'
            params[vk] = vv

        await wal.log_mutation(cypher, params, database=database)
        count += 1

    logger.info('Dumped %d relationships', count)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dump a FalkorDB database to WAL files for backup or migration',
    )
    parser.add_argument('wal_dir', help='Directory to write WAL .jsonl files')
    parser.add_argument('--host', default='localhost', help='FalkorDB host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379, help='FalkorDB port (default: 6379)')
    parser.add_argument(
        '--database', default='default_db', help='Database name to dump (default: default_db)'
    )
    parser.add_argument(
        '--max-events-per-file',
        type=int,
        default=DEFAULT_MAX_EVENTS_PER_FILE,
        help=f'Max events per WAL file (default: {DEFAULT_MAX_EVENTS_PER_FILE})',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    try:
        count = asyncio.run(
            dump_wal(
                host=args.host,
                port=args.port,
                database=args.database,
                wal_dir=args.wal_dir,
                max_events_per_file=args.max_events_per_file,
            )
        )
        print(f'Dumped {count} mutations.')
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    main()
