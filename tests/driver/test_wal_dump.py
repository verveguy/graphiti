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

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False


class TestDumpNodes:
    """Test node dumping to WAL."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dumps_entity_nodes(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        '_id': 1,
                        '_labels': ['Entity', 'Person'],
                        '_props': {
                            'uuid': 'e1',
                            'name': 'Alice',
                            'group_id': 'test',
                            'created_at': '2026-01-01T00:00:00Z',
                            'summary': 'A person',
                            'name_embedding': [0.1, 0.2, 0.3],
                        },
                    },
                ],
                ['_id', '_labels', '_props'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_nodes(mock_driver, wal, 'test_db')
        await wal.close()

        assert count == 1

        wal_files = list(wal_dir.glob('*.jsonl'))
        assert len(wal_files) == 1

        with open(wal_files[0]) as f:
            entry = json.loads(f.readline())

        assert 'Entity:Person' in entry['cypher']
        assert 'MERGE' in entry['cypher']
        assert 'vecf32($name_embedding)' in entry['cypher']
        assert entry['params']['uuid'] == 'e1'
        assert entry['params']['name_embedding'] == [0.1, 0.2, 0.3]
        # name_embedding should NOT be in scalar props
        assert 'name_embedding' not in entry['params']['props']

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dumps_episodic_nodes(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        '_id': 2,
                        '_labels': ['Episodic'],
                        '_props': {
                            'uuid': 'ep1',
                            'name': 'Episode 1',
                            'group_id': 'test',
                            'source': 'text',
                            'content': 'Hello world',
                            'created_at': '2026-01-01T00:00:00Z',
                            'valid_at': '2026-01-01T00:00:00Z',
                        },
                    },
                ],
                ['_id', '_labels', '_props'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_nodes(mock_driver, wal, 'test_db')
        await wal.close()

        assert count == 1

        with open(list(wal_dir.glob('*.jsonl'))[0]) as f:
            entry = json.loads(f.readline())

        assert 'Episodic' in entry['cypher']
        assert 'vecf32' not in entry['cypher']  # No embeddings on episodes

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_skips_nodes_without_uuid(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [{'_id': 1, '_labels': ['Orphan'], '_props': {'name': 'no uuid'}}],
                ['_id', '_labels', '_props'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_nodes(mock_driver, wal, 'db')
        await wal.close()

        assert count == 0

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_empty_database(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(return_value=None)

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_nodes(mock_driver, wal, 'db')
        await wal.close()

        assert count == 0


class TestDumpRelationships:
    """Test relationship dumping to WAL."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dumps_relates_to_edges(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_relationships

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        '_type': 'RELATES_TO',
                        '_src_uuid': 'e1',
                        '_dst_uuid': 'e2',
                        '_props': {
                            'uuid': 'r1',
                            'name': 'knows',
                            'fact': 'Alice knows Bob',
                            'group_id': 'test',
                            'created_at': '2026-01-01T00:00:00Z',
                            'fact_embedding': [0.4, 0.5, 0.6],
                        },
                        '_src_labels': ['Entity'],
                        '_dst_labels': ['Entity'],
                    },
                ],
                ['_type', '_src_uuid', '_dst_uuid', '_props', '_src_labels', '_dst_labels'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_relationships(mock_driver, wal, 'test_db')
        await wal.close()

        assert count == 1

        with open(list(wal_dir.glob('*.jsonl'))[0]) as f:
            entry = json.loads(f.readline())

        assert 'RELATES_TO' in entry['cypher']
        assert 'vecf32($fact_embedding)' in entry['cypher']
        assert entry['params']['src_uuid'] == 'e1'
        assert entry['params']['dst_uuid'] == 'e2'
        assert entry['params']['fact_embedding'] == [0.4, 0.5, 0.6]
        # fact_embedding should NOT be in scalar props
        assert 'fact_embedding' not in entry['params']['props']

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dumps_mentions_edges(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_relationships

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        '_type': 'MENTIONS',
                        '_src_uuid': 'ep1',
                        '_dst_uuid': 'e1',
                        '_props': {
                            'uuid': 'm1',
                            'group_id': 'test',
                            'created_at': '2026-01-01T00:00:00Z',
                        },
                        '_src_labels': ['Episodic'],
                        '_dst_labels': ['Entity'],
                    },
                ],
                ['_type', '_src_uuid', '_dst_uuid', '_props', '_src_labels', '_dst_labels'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_relationships(mock_driver, wal, 'test_db')
        await wal.close()

        assert count == 1

        with open(list(wal_dir.glob('*.jsonl'))[0]) as f:
            entry = json.loads(f.readline())

        assert 'MENTIONS' in entry['cypher']
        assert 'vecf32' not in entry['cypher']  # No embeddings on MENTIONS

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_skips_relationships_without_endpoint_uuids(self, tmp_path):
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_relationships

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        '_type': 'RELATES_TO',
                        '_src_uuid': None,
                        '_dst_uuid': 'e2',
                        '_props': {'uuid': 'r1'},
                        '_src_labels': ['Entity'],
                        '_dst_labels': ['Entity'],
                    },
                ],
                ['_type', '_src_uuid', '_dst_uuid', '_props', '_src_labels', '_dst_labels'],
                None,
            )
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        count = await _dump_relationships(mock_driver, wal, 'db')
        await wal.close()

        assert count == 0


class TestDumpEndToEnd:
    """Test full dump → replay cycle."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dump_then_dry_run_replay(self, tmp_path):
        """Dump mock data, then verify replay can read it."""
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes, _dump_relationships
        from graphiti_core.driver.wal_replay import replay_wal

        mock_driver = MagicMock()

        # First call returns nodes, second returns relationships
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                (
                    [
                        {
                            '_id': 1,
                            '_labels': ['Entity'],
                            '_props': {
                                'uuid': 'e1',
                                'name': 'Alice',
                                'group_id': 'g1',
                                'created_at': '2026-01-01T00:00:00Z',
                            },
                        },
                        {
                            '_id': 2,
                            '_labels': ['Entity'],
                            '_props': {
                                'uuid': 'e2',
                                'name': 'Bob',
                                'group_id': 'g1',
                                'created_at': '2026-01-01T00:00:00Z',
                            },
                        },
                    ],
                    ['_id', '_labels', '_props'],
                    None,
                ),
                (
                    [
                        {
                            '_type': 'RELATES_TO',
                            '_src_uuid': 'e1',
                            '_dst_uuid': 'e2',
                            '_props': {
                                'uuid': 'r1',
                                'name': 'knows',
                                'fact': 'Alice knows Bob',
                                'group_id': 'g1',
                                'created_at': '2026-01-01T00:00:00Z',
                            },
                            '_src_labels': ['Entity'],
                            '_dst_labels': ['Entity'],
                        },
                    ],
                    ['_type', '_src_uuid', '_dst_uuid', '_props', '_src_labels', '_dst_labels'],
                    None,
                ),
            ]
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        await _dump_nodes(mock_driver, wal, 'db')
        await _dump_relationships(mock_driver, wal, 'db')
        await wal.close()

        # Verify replay reads the dumped data
        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 3  # 2 nodes + 1 relationship

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_dump_preserves_node_order_before_edges(self, tmp_path):
        """Verify nodes come before edges in WAL output."""
        from graphiti_core.driver.wal import WalWriter
        from graphiti_core.driver.wal_dump import _dump_nodes, _dump_relationships

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                (
                    [
                        {
                            '_id': 1,
                            '_labels': ['Entity'],
                            '_props': {'uuid': 'e1', 'name': 'A'},
                        },
                    ],
                    ['_id', '_labels', '_props'],
                    None,
                ),
                (
                    [
                        {
                            '_type': 'RELATES_TO',
                            '_src_uuid': 'e1',
                            '_dst_uuid': 'e1',
                            '_props': {'uuid': 'r1'},
                            '_src_labels': ['Entity'],
                            '_dst_labels': ['Entity'],
                        },
                    ],
                    ['_type', '_src_uuid', '_dst_uuid', '_props', '_src_labels', '_dst_labels'],
                    None,
                ),
            ]
        )

        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)

        await _dump_nodes(mock_driver, wal, 'db')
        await _dump_relationships(mock_driver, wal, 'db')
        await wal.close()

        with open(list(wal_dir.glob('*.jsonl'))[0]) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 2
        assert 'MERGE (n:' in entries[0]['cypher']  # Node first
        assert 'MERGE (src)-[r:' in entries[1]['cypher']  # Edge second
