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
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

logger = logging.getLogger(__name__)


def _scan_databases(wal_files: list[Path], default: str) -> set[str]:
    """Scan WAL files for all distinct database names."""
    databases: set[str] = set()
    for wal_file in wal_files:
        with open(wal_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    databases.add(entry.get('db', default))
                except json.JSONDecodeError:
                    continue
    if not databases:
        databases.add(default)
    return databases


async def replay_wal(
    wal_dir: str | Path,
    host: str = 'localhost',
    port: int = 6379,
    database: str = 'default_db',
    from_seq: int = 0,
    dry_run: bool = False,
) -> int:
    """
    Replay WAL files to reconstruct a FalkorDB database.

    Reads all JSONL files from the WAL directory in filename order,
    executes each mutation against the target database, and skips
    events with seq < from_seq for partial replay / resume.

    Args:
        wal_dir: Directory containing WAL .jsonl files.
        host: FalkorDB host.
        port: FalkorDB port.
        database: Target database name. If None, uses the db field from each WAL event.
        from_seq: Skip events with seq < from_seq (for partial replay).
        dry_run: If True, parse and validate but don't execute mutations.

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

    driver: FalkorDriver | None = None
    replayed = 0
    skipped = 0
    errors = 0

    try:
        if not dry_run:
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            # Scan WAL for all distinct databases so we can build indices on each
            databases = _scan_databases(wal_files, default=database)

            # Create driver WITHOUT WAL to avoid re-logging replayed mutations
            driver = FalkorDriver(host=host, port=int(port), database=database)

            # Build indices on every database found in the WAL
            for db_name in databases:
                logger.info('Building indices and constraints on %s...', db_name)
                db_driver = driver if db_name == database else driver.clone(db_name)
                await db_driver.build_indices_and_constraints()

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
                    event_db = entry.get('db', database)

                    if dry_run:
                        logger.debug('DRY RUN seq=%d db=%s: %s', seq, event_db, cypher[:80])
                        replayed += 1
                        continue

                    try:
                        # Use the event's db field to target the correct database
                        assert driver is not None
                        graph = driver._get_graph(event_db)
                        await graph.query(cypher, params)  # type: ignore[reportUnknownMemberType]
                        replayed += 1
                    except Exception as e:
                        logger.error('Replay error at seq=%d: %s\n  %s', seq, e, cypher[:200])
                        errors += 1

    finally:
        if driver is not None:
            await driver.close()

    logger.info(
        'Replay complete: %d replayed, %d skipped (seq < %d), %d errors',
        replayed,
        skipped,
        from_seq,
        errors,
    )
    return replayed


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Replay graphiti WAL files to reconstruct a FalkorDB database',
    )
    parser.add_argument('wal_dir', help='Directory containing WAL .jsonl files')
    parser.add_argument('--host', default='localhost', help='FalkorDB host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379, help='FalkorDB port (default: 6379)')
    parser.add_argument(
        '--database', default='default_db', help='Target database name (default: default_db)'
    )
    parser.add_argument(
        '--from-seq',
        type=int,
        default=0,
        help='Skip events with seq < FROM_SEQ (default: 0)',
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Parse and validate without executing mutations'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    try:
        count = asyncio.run(
            replay_wal(
                wal_dir=args.wal_dir,
                host=args.host,
                port=args.port,
                database=args.database,
                from_seq=args.from_seq,
                dry_run=args.dry_run,
            )
        )
        print(f'Replayed {count} mutations.')
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    main()
