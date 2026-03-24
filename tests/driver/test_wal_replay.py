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

import pytest

from graphiti_core.driver.wal_replay import replay_wal


def _write_wal_file(wal_dir, filename, entries):
    """Helper to write a JSONL WAL file."""
    path = wal_dir / filename
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return path


class TestReplayWalDryRun:
    """Test WAL replay in dry-run mode (no FalkorDB dependency)."""

    @pytest.mark.asyncio
    async def test_dry_run_counts_mutations(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [
                {
                    'seq': 0,
                    'ts': '2026-03-24T00:00:00Z',
                    'db': 'test_db',
                    'cypher': 'CREATE (n:Test)',
                    'params': {},
                },
                {
                    'seq': 1,
                    'ts': '2026-03-24T00:00:01Z',
                    'db': 'test_db',
                    'cypher': 'MERGE (n:Entity {uuid: $uuid})',
                    'params': {'uuid': 'abc'},
                },
            ],
        )

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 2

    @pytest.mark.asyncio
    async def test_dry_run_respects_from_seq(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [
                {
                    'seq': 0,
                    'ts': '2026-03-24T00:00:00Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:First)',
                    'params': {},
                },
                {
                    'seq': 1,
                    'ts': '2026-03-24T00:00:01Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:Second)',
                    'params': {},
                },
                {
                    'seq': 2,
                    'ts': '2026-03-24T00:00:02Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:Third)',
                    'params': {},
                },
            ],
        )

        count = await replay_wal(wal_dir, from_seq=2, dry_run=True)
        assert count == 1

    @pytest.mark.asyncio
    async def test_dry_run_empty_directory(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 0

    @pytest.mark.asyncio
    async def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await replay_wal(tmp_path / 'nonexistent', dry_run=True)

    @pytest.mark.asyncio
    async def test_dry_run_multiple_files_in_order(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [
                {
                    'seq': 0,
                    'ts': '2026-03-24T00:00:00Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:File1)',
                    'params': {},
                },
            ],
        )
        _write_wal_file(
            wal_dir,
            '20260324_000100_abc123_0001.jsonl',
            [
                {
                    'seq': 1,
                    'ts': '2026-03-24T00:01:00Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:File2)',
                    'params': {},
                },
                {
                    'seq': 2,
                    'ts': '2026-03-24T00:01:01Z',
                    'db': 'db',
                    'cypher': 'CREATE (n:File2b)',
                    'params': {},
                },
            ],
        )

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 3

    @pytest.mark.asyncio
    async def test_dry_run_skips_blank_lines(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        path = wal_dir / '20260324_000000_abc123_0000.jsonl'
        with open(path, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'seq': 0,
                        'ts': '2026-03-24T00:00:00Z',
                        'db': 'db',
                        'cypher': 'CREATE (n:Test)',
                        'params': {},
                    }
                )
                + '\n'
            )
            f.write('\n')  # blank line
            f.write('\n')  # another blank line
            f.write(
                json.dumps(
                    {
                        'seq': 1,
                        'ts': '2026-03-24T00:00:01Z',
                        'db': 'db',
                        'cypher': 'CREATE (n:Test2)',
                        'params': {},
                    }
                )
                + '\n'
            )

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 2

    @pytest.mark.asyncio
    async def test_dry_run_handles_malformed_json(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        path = wal_dir / '20260324_000000_abc123_0000.jsonl'
        with open(path, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'seq': 0,
                        'ts': '2026-03-24T00:00:00Z',
                        'db': 'db',
                        'cypher': 'CREATE (n:Good)',
                        'params': {},
                    }
                )
                + '\n'
            )
            f.write('{bad json\n')  # malformed
            f.write(
                json.dumps(
                    {
                        'seq': 2,
                        'ts': '2026-03-24T00:00:02Z',
                        'db': 'db',
                        'cypher': 'CREATE (n:AlsoGood)',
                        'params': {},
                    }
                )
                + '\n'
            )

        # Should replay the valid entries and skip the bad one
        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 2


class TestReplayWalEndToEnd:
    """Test full WAL write → replay cycle using WalWriter + dry-run replay."""

    @pytest.mark.asyncio
    async def test_write_then_replay_dry_run(self, tmp_path):
        """Write mutations via WalWriter, then verify replay reads them correctly."""
        from graphiti_core.driver.wal import WalWriter

        wal_dir = tmp_path / 'wal'
        writer = WalWriter(wal_dir)

        await writer.log_mutation(
            'MERGE (n:Entity {uuid: $uuid}) SET n.name = $name',
            {'uuid': 'e1', 'name': 'Alice'},
            database='mydb',
        )
        await writer.log_mutation(
            'MERGE (n:Entity {uuid: $uuid}) SET n.name = $name',
            {'uuid': 'e2', 'name': 'Bob'},
            database='mydb',
        )
        # This should NOT appear in WAL (read-only)
        await writer.log_mutation('MATCH (n) RETURN n', {}, database='mydb')
        await writer.close()

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 2

    @pytest.mark.asyncio
    async def test_write_rotate_then_replay(self, tmp_path):
        """Write enough to trigger rotation, then replay all files."""
        from graphiti_core.driver.wal import WalWriter

        wal_dir = tmp_path / 'wal'
        writer = WalWriter(wal_dir, max_events_per_file=3)

        for i in range(7):
            await writer.log_mutation(
                f'CREATE (n:Node{i})', {}, database='db'
            )
        await writer.close()

        # Should have 3 files: 3 + 3 + 1
        wal_files = sorted(wal_dir.glob('*.jsonl'))
        assert len(wal_files) == 3

        count = await replay_wal(wal_dir, dry_run=True)
        assert count == 7

    @pytest.mark.asyncio
    async def test_partial_replay_with_from_seq(self, tmp_path):
        """Write mutations, then replay only from a specific sequence."""
        from graphiti_core.driver.wal import WalWriter

        wal_dir = tmp_path / 'wal'
        writer = WalWriter(wal_dir)

        for i in range(5):
            await writer.log_mutation(f'CREATE (n:Node{i})', {}, database='db')
        await writer.close()

        # Replay only seq >= 3
        count = await replay_wal(wal_dir, from_seq=3, dry_run=True)
        assert count == 2
