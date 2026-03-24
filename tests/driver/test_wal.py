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
from datetime import datetime, timezone

import pytest

from graphiti_core.driver.wal import WalWriter


class TestIsMutation:
    """Test WalWriter.is_mutation() query classification."""

    def test_match_only_is_not_mutation(self):
        """MATCH-only queries are reads, not mutations."""
        assert WalWriter.is_mutation('MATCH (n) RETURN n') is False
        assert WalWriter.is_mutation('MATCH (n:Person) WHERE n.name = "test" RETURN n') is False

    def test_create_is_mutation(self):
        """CREATE queries are mutations."""
        assert WalWriter.is_mutation('CREATE (n:Person {name: "test"})') is True
        assert WalWriter.is_mutation('MATCH (a) CREATE (b)-[:KNOWS]->(a)') is True

    def test_merge_is_mutation(self):
        """MERGE queries are mutations."""
        assert WalWriter.is_mutation('MERGE (n:Person {name: "test"})') is True
        assert WalWriter.is_mutation('MATCH (a) MERGE (a)-[:KNOWS]->(b)') is True

    def test_set_is_mutation(self):
        """SET queries are mutations."""
        assert WalWriter.is_mutation('MATCH (n) SET n.name = "test"') is True
        assert WalWriter.is_mutation('MATCH (n) SET n += {age: 30}') is True

    def test_delete_is_mutation(self):
        """DELETE queries are mutations."""
        assert WalWriter.is_mutation('MATCH (n) DELETE n') is True
        assert WalWriter.is_mutation('MATCH (n:Person) WHERE n.age < 18 DELETE n') is True

    def test_detach_delete_is_mutation(self):
        """DETACH DELETE queries are mutations."""
        assert WalWriter.is_mutation('MATCH (n) DETACH DELETE n') is True

    def test_drop_is_mutation(self):
        """DROP queries are mutations."""
        assert WalWriter.is_mutation('DROP INDEX idx_name') is True

    def test_remove_is_mutation(self):
        """REMOVE queries are mutations."""
        assert WalWriter.is_mutation('MATCH (n) REMOVE n.name') is True
        assert WalWriter.is_mutation('MATCH (n) REMOVE n:Label') is True

    def test_keyword_in_string_is_not_mutation(self):
        """Keywords inside string literals should not trigger false positives."""
        assert WalWriter.is_mutation("MATCH (n) WHERE n.name = 'CREATE'") is False
        assert WalWriter.is_mutation('MATCH (n) WHERE n.action = "DELETE"') is False
        assert WalWriter.is_mutation("MATCH (n) WHERE n.data = 'SET value'") is False

    def test_case_insensitive_detection(self):
        """Mutation keywords should be detected case-insensitively."""
        assert WalWriter.is_mutation('create (n:Test)') is True
        assert WalWriter.is_mutation('Create (n:Test)') is True
        assert WalWriter.is_mutation('MATCH (n) set n.x = 1') is True
        assert WalWriter.is_mutation('match (n) Delete n') is True

    def test_complex_queries(self):
        """Test complex multi-clause queries."""
        # MATCH + CREATE
        assert WalWriter.is_mutation('MATCH (a:Person) CREATE (a)-[:KNOWS]->(b:Person)') is True
        # MATCH + MERGE + SET
        assert (
            WalWriter.is_mutation('MATCH (n) MERGE (m:Test) ON CREATE SET m.created = true') is True
        )
        # Multiple operations
        assert (
            WalWriter.is_mutation(
                'MATCH (a), (b) WHERE a.id = 1 AND b.id = 2 CREATE (a)-[:REL]->(b) SET a.linked = true'
            )
            is True
        )


class TestIsIndexDDL:
    """Test WalWriter.is_index_ddl() index detection."""

    def test_create_index_is_ddl(self):
        """CREATE INDEX queries are DDL."""
        assert WalWriter.is_index_ddl('CREATE INDEX FOR (n:Person) ON (n.name)') is True
        assert WalWriter.is_index_ddl('CREATE INDEX idx_name FOR (n:Person) ON (n.name)') is True

    def test_create_unique_index_is_ddl(self):
        """CREATE UNIQUE INDEX queries are DDL."""
        assert WalWriter.is_index_ddl('CREATE UNIQUE INDEX FOR (n:Person) ON (n.id)') is True

    def test_create_fulltext_index_is_ddl(self):
        """CREATE FULLTEXT INDEX queries are DDL."""
        assert (
            WalWriter.is_index_ddl('CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name, n.bio)')
            is True
        )

    def test_create_vector_index_is_ddl(self):
        """CREATE VECTOR INDEX queries are DDL."""
        assert WalWriter.is_index_ddl('CREATE VECTOR INDEX FOR (n:Person) ON (n.embedding)') is True

    def test_drop_index_is_ddl(self):
        """DROP INDEX queries are DDL."""
        assert WalWriter.is_index_ddl('DROP INDEX idx_name') is True
        assert WalWriter.is_index_ddl('DROP INDEX ON :Person(name)') is True

    def test_drop_fulltext_index_is_ddl(self):
        """DROP FULLTEXT INDEX queries are DDL."""
        assert WalWriter.is_index_ddl('DROP FULLTEXT INDEX FOR (n:Person) ON (n.name)') is True

    def test_call_db_idx_fulltext_is_ddl(self):
        """CALL db.idx.fulltext queries are DDL."""
        assert (
            WalWriter.is_index_ddl("CALL db.idx.fulltext.createNodeIndex('Person', 'name')") is True
        )

    def test_call_db_indexes_is_ddl(self):
        """CALL db.indexes() queries are DDL."""
        assert WalWriter.is_index_ddl('CALL db.indexes()') is True

    def test_regular_mutations_are_not_ddl(self):
        """Regular mutation queries are not DDL."""
        assert WalWriter.is_index_ddl('CREATE (n:Person {name: "test"})') is False
        assert WalWriter.is_index_ddl('MERGE (n:Person {id: 1})') is False
        assert WalWriter.is_index_ddl('MATCH (n) DELETE n') is False

    def test_case_insensitive_detection(self):
        """Index DDL should be detected case-insensitively."""
        assert WalWriter.is_index_ddl('create index FOR (n:Test) ON (n.id)') is True
        assert WalWriter.is_index_ddl('Create Fulltext Index FOR (n:Test) ON (n.name)') is True
        assert WalWriter.is_index_ddl('drop index idx_name') is True


class TestWalWriter:
    """Test WalWriter file operations, rotation, and sequence handling."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        """Create a temporary WAL directory."""
        return tmp_path / 'wal'

    @pytest.fixture
    def wal_writer(self, wal_dir):
        """Create a WalWriter instance."""
        writer = WalWriter(wal_dir, max_events_per_file=5)
        yield writer
        # Cleanup - close if not already closed
        asyncio.get_event_loop().run_until_complete(writer.close())

    @pytest.mark.asyncio
    async def test_creates_wal_directory(self, wal_dir):
        """WalWriter creates WAL directory if it doesn't exist."""
        assert not wal_dir.exists()
        writer = WalWriter(wal_dir)
        assert wal_dir.exists()
        await writer.close()

    @pytest.mark.asyncio
    async def test_log_mutation_creates_file(self, wal_dir):
        """First logged mutation creates a WAL file."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test)', {}, 'test_db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_log_mutation_writes_jsonl_entry(self, wal_dir):
        """Logged mutations are written as JSONL entries."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test {name: $name})', {'name': 'test'}, 'test_db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            entry = json.loads(f.readline())

        assert entry['seq'] == 0
        assert 'ts' in entry
        assert entry['db'] == 'test_db'
        assert entry['cypher'] == 'CREATE (n:Test {name: $name})'
        assert entry['params'] == {'name': 'test'}

    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self, wal_dir):
        """Sequence numbers increment with each logged mutation."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test1)', {}, 'db')
        await writer.log_mutation('CREATE (n:Test2)', {}, 'db')
        await writer.log_mutation('CREATE (n:Test3)', {}, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0])['seq'] == 0
        assert json.loads(lines[1])['seq'] == 1
        assert json.loads(lines[2])['seq'] == 2

    @pytest.mark.asyncio
    async def test_file_rotation(self, wal_dir):
        """Files rotate after max_events_per_file entries."""
        writer = WalWriter(wal_dir, max_events_per_file=3)

        # Write 5 mutations - should create 2 files (3 + 2)
        for i in range(5):
            await writer.log_mutation(f'CREATE (n:Test{i})', {}, 'db')

        await writer.close()

        files = sorted(wal_dir.glob('*.jsonl'))
        assert len(files) == 2

        # First file should have 3 entries
        with open(files[0]) as f:
            assert len(f.readlines()) == 3

        # Second file should have 2 entries
        with open(files[1]) as f:
            assert len(f.readlines()) == 2

    @pytest.mark.asyncio
    async def test_sequence_continuity_across_files(self, wal_dir):
        """Sequence numbers continue across file boundaries."""
        writer = WalWriter(wal_dir, max_events_per_file=2)

        for i in range(4):
            await writer.log_mutation(f'CREATE (n:Test{i})', {}, 'db')

        await writer.close()

        # Read all entries from all files
        all_seqs = []
        for f in sorted(wal_dir.glob('*.jsonl')):
            with open(f) as fp:
                for line in fp:
                    all_seqs.append(json.loads(line)['seq'])

        assert all_seqs == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_sequence_continuity_on_restart(self, wal_dir):
        """Sequence numbers continue from where previous session left off."""
        # First session
        writer1 = WalWriter(wal_dir)
        await writer1.log_mutation('CREATE (n:Test1)', {}, 'db')
        await writer1.log_mutation('CREATE (n:Test2)', {}, 'db')
        await writer1.close()

        # Second session - should continue sequence
        writer2 = WalWriter(wal_dir)
        assert writer2.current_sequence == 2  # Should resume from where first left off
        await writer2.log_mutation('CREATE (n:Test3)', {}, 'db')
        await writer2.close()

        # Check sequences across all files (order doesn't matter, just that all seqs present)
        all_seqs = []
        for f in wal_dir.glob('*.jsonl'):
            with open(f) as fp:
                for line in fp:
                    all_seqs.append(json.loads(line)['seq'])

        assert sorted(all_seqs) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_filters_read_only_queries(self, wal_dir):
        """Read-only queries are not logged."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('MATCH (n) RETURN n', {}, 'db')
        await writer.log_mutation('MATCH (n:Person) WHERE n.age > 18 RETURN n', {}, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 0  # No file created since nothing was logged

    @pytest.mark.asyncio
    async def test_filters_index_ddl(self, wal_dir):
        """Index DDL queries are not logged."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE INDEX FOR (n:Person) ON (n.name)', {}, 'db')
        await writer.log_mutation('DROP INDEX idx_name', {}, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_serializes_datetime_params(self, wal_dir):
        """Datetime parameters are serialized to ISO format."""
        writer = WalWriter(wal_dir)
        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        await writer.log_mutation('CREATE (n:Test {created: $ts})', {'ts': dt}, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            entry = json.loads(f.readline())

        assert entry['params']['ts'] == '2024-01-15T12:30:45+00:00'

    @pytest.mark.asyncio
    async def test_serializes_nested_datetime_params(self, wal_dir):
        """Nested datetime parameters are serialized correctly."""
        writer = WalWriter(wal_dir)
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        params = {
            'data': {'created': dt, 'items': [dt, dt]},
            'tuple_data': (dt, 'string'),
        }
        await writer.log_mutation('CREATE (n:Test $props)', params, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            entry = json.loads(f.readline())

        assert entry['params']['data']['created'] == '2024-01-15T00:00:00+00:00'
        assert entry['params']['data']['items'] == [
            '2024-01-15T00:00:00+00:00',
            '2024-01-15T00:00:00+00:00',
        ]
        # Tuples are converted to lists in JSON
        assert entry['params']['tuple_data'] == ['2024-01-15T00:00:00+00:00', 'string']

    @pytest.mark.asyncio
    async def test_serializes_embedding_params(self, wal_dir):
        """Embedding (list of floats) parameters are serialized correctly."""
        writer = WalWriter(wal_dir)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        await writer.log_mutation(
            'CREATE (n:Test {embedding: $emb})', {'emb': embedding, 'name': 'test'}, 'db'
        )
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            entry = json.loads(f.readline())

        assert entry['params']['emb'] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert entry['params']['name'] == 'test'

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, wal_dir):
        """Calling close() multiple times is safe."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test)', {}, 'db')

        # Multiple closes should not raise
        await writer.close()
        await writer.close()
        await writer.close()

        assert writer.is_closed is True

    @pytest.mark.asyncio
    async def test_log_after_close_is_ignored(self, wal_dir):
        """Logging after close is ignored with a warning."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test1)', {}, 'db')
        await writer.close()

        # This should be silently ignored
        await writer.log_mutation('CREATE (n:Test2)', {}, 'db')

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_current_sequence_property(self, wal_dir):
        """current_sequence property returns next sequence to be written."""
        writer = WalWriter(wal_dir)
        assert writer.current_sequence == 0

        await writer.log_mutation('CREATE (n:Test1)', {}, 'db')
        assert writer.current_sequence == 1

        await writer.log_mutation('CREATE (n:Test2)', {}, 'db')
        assert writer.current_sequence == 2

        await writer.close()



class TestWalWriterConcurrency:
    """Test concurrent write handling."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        """Create a temporary WAL directory."""
        return tmp_path / 'wal'

    @pytest.mark.asyncio
    async def test_concurrent_writes_dont_interleave(self, wal_dir):
        """Concurrent writes produce valid JSONL (no interleaved lines)."""
        writer = WalWriter(wal_dir, max_events_per_file=100)

        async def write_batch(start: int, count: int):
            for i in range(count):
                await writer.log_mutation(
                    f'CREATE (n:Test {{id: {start + i}}})', {'idx': start + i}, 'db'
                )

        # Launch concurrent writers
        await asyncio.gather(
            write_batch(0, 10),
            write_batch(100, 10),
            write_batch(200, 10),
        )

        await writer.close()

        # Verify all entries are valid JSON
        entries = []
        for f in wal_dir.glob('*.jsonl'):
            with open(f) as fp:
                for line in fp:
                    entry = json.loads(line)  # Should not raise
                    entries.append(entry)

        # Verify all 30 entries were written
        assert len(entries) == 30

        # Verify sequence numbers are unique and monotonic
        seqs = [e['seq'] for e in entries]
        assert sorted(seqs) == list(range(30))

    @pytest.mark.asyncio
    async def test_concurrent_writes_with_rotation(self, wal_dir):
        """Concurrent writes with file rotation maintain sequence integrity."""
        writer = WalWriter(wal_dir, max_events_per_file=5)

        async def write_batch(start: int, count: int):
            for i in range(count):
                await writer.log_mutation(
                    f'CREATE (n:Test {{id: {start + i}}})', {'idx': start + i}, 'db'
                )

        await asyncio.gather(
            write_batch(0, 8),
            write_batch(100, 8),
        )

        await writer.close()

        # Read all entries
        entries = []
        for f in sorted(wal_dir.glob('*.jsonl')):
            with open(f) as fp:
                for line in fp:
                    entries.append(json.loads(line))

        # Verify all 16 entries
        assert len(entries) == 16

        # Verify sequences are unique and ordered
        seqs = [e['seq'] for e in entries]
        assert sorted(seqs) == list(range(16))


class TestWalWriterRefCounting:
    """Test reference counting for shared WAL instances."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        """Create a temporary WAL directory."""
        return tmp_path / 'wal'

    @pytest.mark.asyncio
    async def test_close_clone_does_not_close_shared_wal(self, wal_dir):
        """Closing a clone's reference should not close the shared WAL."""
        writer = WalWriter(wal_dir)
        writer.acquire()  # Simulate clone taking a reference

        # Clone closes its reference
        await writer.close()
        assert writer.is_closed is False

        # Original can still log
        await writer.log_mutation('CREATE (n:Test)', {}, 'db')
        assert writer.current_sequence == 1

        # Original closes — now it's truly closed
        await writer.close()
        assert writer.is_closed is True

    @pytest.mark.asyncio
    async def test_multiple_acquires_require_matching_closes(self, wal_dir):
        """Each acquire() requires a matching close() before the WAL shuts down."""
        writer = WalWriter(wal_dir)
        writer.acquire()
        writer.acquire()

        await writer.close()
        assert writer.is_closed is False

        await writer.close()
        assert writer.is_closed is False

        await writer.close()
        assert writer.is_closed is True

    @pytest.mark.asyncio
    async def test_acquire_after_close_raises(self, wal_dir):
        """Cannot acquire a reference on an already-closed WAL."""
        writer = WalWriter(wal_dir)
        await writer.close()

        with pytest.raises(RuntimeError, match='Cannot acquire a closed WAL writer'):
            writer.acquire()

    @pytest.mark.asyncio
    async def test_extra_close_after_shutdown_is_noop(self, wal_dir):
        """Extra close() calls after the WAL is shut down are safe no-ops."""
        writer = WalWriter(wal_dir)
        await writer.close()
        assert writer.is_closed is True

        # Should not raise
        await writer.close()
        await writer.close()
        assert writer.is_closed is True

    @pytest.mark.asyncio
    async def test_mutations_work_until_last_close(self, wal_dir):
        """Mutations can be logged as long as at least one reference remains."""
        writer = WalWriter(wal_dir)
        writer.acquire()  # ref_count = 2

        await writer.log_mutation('CREATE (n:A)', {}, 'db')
        await writer.close()  # ref_count = 1, still open

        await writer.log_mutation('CREATE (n:B)', {}, 'db')
        await writer.close()  # ref_count = 0, closed

        # After final close, mutation is dropped
        await writer.log_mutation('CREATE (n:C)', {}, 'db')

        files = list(wal_dir.glob('*.jsonl'))
        with open(files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])['cypher'] == 'CREATE (n:A)'
        assert json.loads(lines[1])['cypher'] == 'CREATE (n:B)'
