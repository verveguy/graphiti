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

import pytest

from graphiti_core.driver.ladybug_driver import LadybugDriver
from graphiti_core.driver.wal import WalWriter


class TestLadybugDriverClose:
    """Verify that LadybugDriver.close() correctly tears down the DB/connection."""

    @pytest.mark.asyncio
    async def test_close_file_backed_does_not_raise(self, tmp_path):
        """close() on a file-backed database must not raise."""
        db_path = str(tmp_path / 'test.db')
        driver = LadybugDriver(db=db_path)
        # Should complete without error and checkpoint the Kuzu WAL.
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_idempotent_file_backed(self, tmp_path):
        """Calling close() twice on a file-backed database must not raise."""
        db_path = str(tmp_path / 'test.db')
        driver = LadybugDriver(db=db_path)
        await driver.close()
        # Second call must also succeed — guarded by the underlying library's
        # own idempotency (verified during research).
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_in_memory_does_not_raise(self):
        """close() on an in-memory database must not raise."""
        driver = LadybugDriver(db=':memory:')
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_with_wal_does_not_raise(self, tmp_path):
        """close() flushes the JSONL WAL and closes the DB without error."""
        db_path = str(tmp_path / 'test.db')
        wal_dir = tmp_path / 'wal'
        driver = LadybugDriver(db=db_path, wal_dir=wal_dir)
        await driver.close()


class TestLadybugDriverWalChunk:
    """Verify driver-level wal_chunk() batches WAL mutations per chunk context.

    These tests interact with the WAL writer directly (via driver._wal) rather
    than through execute_query() to avoid a pre-existing LadybugDB segfault in
    query result __del__ when the DB is closed before GC runs. The integration
    contract being verified is that driver.wal_chunk() correctly delegates to
    WalWriter.chunk() and that the chunk semantics (single file, discard on
    exception) hold through the driver layer.
    """

    @pytest.mark.asyncio
    async def test_chunk_produces_one_file(self, tmp_path):
        """5 mutations inside wal_chunk() flush to exactly 1 JSONL file."""
        db_path = str(tmp_path / 'test.db')
        wal_dir = tmp_path / 'wal'
        # Use max_events_per_file=2 to confirm rotation is bypassed inside chunk
        wal = WalWriter(wal_dir, max_events_per_file=2)
        driver = LadybugDriver(db=db_path, wal_writer=wal)

        async with driver.wal_chunk():
            for i in range(5):
                await driver._wal.log_mutation(f'CREATE (n:Test{i})', {}, '')

        await driver.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 1
        with open(files[0]) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_non_chunk_produces_one_file_per_mutation(self, tmp_path):
        """3 mutations outside any chunk each go to separate files (max_events=1)."""
        db_path = str(tmp_path / 'test.db')
        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir, max_events_per_file=1)
        driver = LadybugDriver(db=db_path, wal_writer=wal)

        for i in range(3):
            await driver._wal.log_mutation(f'CREATE (n:Test{i})', {}, '')

        await driver.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_chunk_exception_writes_no_file(self, tmp_path):
        """Exception inside wal_chunk() discards the buffer — no file is written."""
        db_path = str(tmp_path / 'test.db')
        wal_dir = tmp_path / 'wal'
        wal = WalWriter(wal_dir)
        driver = LadybugDriver(db=db_path, wal_writer=wal)

        with pytest.raises(RuntimeError, match='simulated'):
            async with driver.wal_chunk():
                await driver._wal.log_mutation('CREATE (n:Test)', {}, '')
                raise RuntimeError('simulated chunk failure')

        await driver.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_wal_chunk_noop_when_wal_disabled(self, tmp_path):
        """wal_chunk() returns nullcontext when WAL is disabled — no error raised."""
        db_path = str(tmp_path / 'test.db')
        driver = LadybugDriver(db=db_path)  # no wal_dir

        # Should not raise even though no WAL is configured
        async with driver.wal_chunk():
            pass  # no-op context

        await driver.close()
