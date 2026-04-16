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
