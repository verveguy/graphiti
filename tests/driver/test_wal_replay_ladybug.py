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

import asyncio
import json

import pytest

from graphiti_core.driver.ladybug_driver import replay_wal_ladybug


def _write_wal_file(wal_dir, filename, entries):
    """Write a JSONL WAL file with the given entries."""
    path = wal_dir / filename
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return path


def _make_entry(seq):
    return {
        'seq': seq,
        'ts': '2026-03-24T00:00:00Z',
        'db': 'testdb',
        'cypher': f'CREATE (n:Node{seq})',
        'params': {},
    }


class TestReplayWalLadybug:
    """Tests for replay_wal_ladybug() using dry_run mode (no DB dependency)."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_correct_count(self, tmp_path):
        """dry_run replay of 5 WAL entries must return exactly 5."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [_make_entry(i) for i in range(5)],
        )

        count = await replay_wal_ladybug(wal_dir, db=':memory:', dry_run=True)
        assert count == 5

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_dry_run_does_not_block_event_loop(self, tmp_path):
        """The event loop must remain schedulable while replay runs.

        A concurrent coroutine that increments a counter with asyncio.sleep()
        should advance during replay — not be starved until replay completes.
        """
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [_make_entry(i) for i in range(50_000)],
        )

        counter = {'value': 0}
        done = asyncio.Event()

        async def increment_while_running():
            while not done.is_set():
                counter['value'] += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(increment_while_running())
        try:
            count = await replay_wal_ladybug(wal_dir, db=':memory:', dry_run=True)
        finally:
            done.set()
            await task

        assert count == 50_000
        assert counter['value'] > 0, (
            'Counter never advanced during replay — event loop was blocked'
        )

    @pytest.mark.asyncio
    async def test_dry_run_empty_directory_returns_zero(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        count = await replay_wal_ladybug(wal_dir, db=':memory:', dry_run=True)
        assert count == 0

    @pytest.mark.asyncio
    async def test_dry_run_respects_from_seq(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(
            wal_dir,
            '20260324_000000_abc123_0000.jsonl',
            [_make_entry(i) for i in range(5)],
        )

        count = await replay_wal_ladybug(wal_dir, db=':memory:', from_seq=3, dry_run=True)
        assert count == 2  # seq 3 and 4

    @pytest.mark.asyncio
    async def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await replay_wal_ladybug(tmp_path / 'nonexistent', db=':memory:', dry_run=True)
