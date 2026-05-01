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
from graphiti_core.driver.wal_replay_helpers import decode_embedding_param


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
    async def wal_writer(self, wal_dir):
        """Create a WalWriter instance."""
        writer = WalWriter(wal_dir, max_events_per_file=5)
        yield writer
        # Cleanup - close if not already closed
        await writer.close()

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
    async def test_scan_handles_chunk_boundary_in_utf8_continuation_byte(self, wal_dir):
        """Backward chunk-scan tolerates UnicodeDecodeError on partial-line UTF-8.

        When a WAL line is longer than the 8KB chunk size used by
        `_scan_existing_files`, the leading edge of the first chunk read
        starts mid-line. If that mid-line position lands inside a UTF-8
        multibyte sequence, json.loads raises UnicodeDecodeError before
        even reaching the JSON parser. The scan must skip these partial
        lines, not abort.

        Regression test: previously, only json.JSONDecodeError was caught
        and any 0x80-0xBF byte at the chunk boundary blew up service init.
        """
        # Build a single WAL line longer than one 8KB chunk (~12KB).
        # Place a UTF-8 continuation byte at the position where the
        # backward chunk-read boundary lands (file_size - 8192).
        seq = 42
        prefix = json.dumps(
            {
                'seq': seq,
                'ts': 't',
                'db': 'd',
                'cypher': 'CREATE (n:Big)',
                'params': {'x': 'A' * 8000},
            },
            separators=(',', ':'),
        )
        # `prefix` is ASCII; pad with bytes that include a 0xa0 at the
        # known chunk boundary. We synthesize a fake WAL file directly
        # rather than going through log_mutation since we want byte-level
        # control of the layout.
        wal_path = wal_dir / '20260101_000000_abc123_0000.jsonl'

        # Construct: a long line whose byte at offset (line_len - 8192)
        # equals 0xa0. We can't easily inject 0xa0 into a JSON string
        # since json.loads would reject it — but we don't need to: a
        # short well-formed first line (which is what _scan_existing_files
        # ultimately returns), preceded by a byte 0xa0 sitting at the
        # *first* chunk boundary. Build with raw bytes.
        well_formed_tail = (
            json.dumps({'seq': seq, 'ts': 't', 'db': 'd', 'cypher': 'X', 'params': {}}) + '\n'
        ).encode('utf-8')
        # Pad to push tail past the 8192-byte boundary, with byte 0xa0 sitting
        # exactly where the backward chunk-read first lands.
        pad_len = 9000
        body = b'\xa0' + b'X' * (pad_len - 1) + b'\n' + well_formed_tail

        wal_path.write_bytes(body)

        # If the scan fails on the chunk boundary, this raises
        # UnicodeDecodeError. With the fix, it should resume from seq 42.
        writer = WalWriter(wal_dir)
        assert writer.current_sequence == seq + 1
        await writer.close()

    @pytest.mark.asyncio
    async def test_scan_handles_non_dict_json_value(self, wal_dir):
        """Backward chunk-scan tolerates JSON lines that decode to non-dict.

        json.loads("null") returns None; `'seq' in None` would TypeError.
        The scan must filter to dict-shaped entries and keep walking back.
        """
        wal_path = wal_dir / '20260101_000000_abc123_0000.jsonl'
        wal_path.write_bytes(
            b'null\n[1, 2, 3]\n42\n' + json.dumps({'seq': 7, 'ts': 't', 'db': 'd'}).encode() + b'\n'
        )
        writer = WalWriter(wal_dir)
        assert writer.current_sequence == 8
        await writer.close()

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

    @pytest.mark.asyncio
    async def test_rotate_closes_file_and_increments_seq(self, wal_dir):
        """rotate() closes the current file; next write opens a new one."""
        writer = WalWriter(wal_dir, max_events_per_file=100)

        # Write one mutation to open a file
        await writer.log_mutation('CREATE (n:Test1)', {}, 'db')
        assert writer._file is not None
        initial_file_seq = writer._file_seq

        # Rotate
        await writer.rotate()
        assert writer._file is None
        assert writer._file_seq == initial_file_seq + 1

        # Write another mutation — should open a new file
        await writer.log_mutation('CREATE (n:Test2)', {}, 'db')
        assert writer._file is not None

        await writer.close()

        # Should have 2 files: one from before rotate, one from after
        files = sorted(wal_dir.glob('*.jsonl'))
        assert len(files) == 2

        # First file has 1 entry, second file has 1 entry
        with open(files[0]) as f:
            assert len(f.readlines()) == 1
        with open(files[1]) as f:
            assert len(f.readlines()) == 1

    @pytest.mark.asyncio
    async def test_rotate_noop_when_no_file_open(self, wal_dir):
        """rotate() is a no-op when no file is open."""
        writer = WalWriter(wal_dir)
        initial_file_seq = writer._file_seq

        # Rotate with no open file — should not raise or change state
        await writer.rotate()
        assert writer._file_seq == initial_file_seq
        assert writer._file is None

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


class TestChunkBatching:
    """Test WalWriter.chunk() context manager for chunk-level WAL batching."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        return tmp_path / 'wal'

    @pytest.mark.asyncio
    async def test_chunk_flush_creates_single_file(self, wal_dir):
        """10 mutations inside a chunk context produce exactly 1 JSONL file with 10 lines."""
        writer = WalWriter(wal_dir, max_events_per_file=3)  # low limit to prove no rotation
        async with writer.chunk():
            for i in range(10):
                await writer.log_mutation(f'CREATE (n:Test{i})', {}, 'db')
        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 1

        with open(files[0]) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 10
        entries = [json.loads(line) for line in lines]
        seqs = [e['seq'] for e in entries]
        assert seqs == list(range(10))

    @pytest.mark.asyncio
    async def test_chunk_exception_discards_buffer(self, wal_dir):
        """Exception inside chunk context discards the buffer — no file is written."""
        writer = WalWriter(wal_dir)

        with pytest.raises(ValueError):
            async with writer.chunk():
                await writer.log_mutation('CREATE (n:Test)', {}, 'db')
                raise ValueError('simulated failure')

        await writer.close()

        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_non_chunk_writes_immediate(self, wal_dir):
        """Mutations outside a chunk context are written immediately to disk."""
        writer = WalWriter(wal_dir)
        await writer.log_mutation('CREATE (n:Test)', {}, 'db')

        # File must exist before any chunk is ever started
        files = list(wal_dir.glob('*.jsonl'))
        assert len(files) == 1

        await writer.close()

    @pytest.mark.asyncio
    async def test_sequence_monotonic_across_chunks(self, wal_dir):
        """Sequence numbers in the second chunk are strictly higher than the first chunk."""
        writer = WalWriter(wal_dir)

        async with writer.chunk():
            for _ in range(3):
                await writer.log_mutation('CREATE (n:A)', {}, 'db')

        async with writer.chunk():
            for _ in range(3):
                await writer.log_mutation('CREATE (n:B)', {}, 'db')

        await writer.close()

        files = sorted(wal_dir.glob('*.jsonl'))
        assert len(files) == 2

        def read_seqs(path):
            with open(path) as f:
                return [json.loads(line)['seq'] for line in f if line.strip()]

        seqs1 = read_seqs(files[0])
        seqs2 = read_seqs(files[1])
        assert seqs1 == [0, 1, 2]
        assert seqs2 == [3, 4, 5]
        assert max(seqs1) < min(seqs2)

    @pytest.mark.asyncio
    async def test_nested_chunk_raises_runtime_error(self, wal_dir):
        """Starting a chunk inside an active chunk raises RuntimeError."""
        writer = WalWriter(wal_dir)

        with pytest.raises(RuntimeError, match='Cannot start a chunk'):
            async with writer.chunk():
                async with writer.chunk():
                    pass

        await writer.close()

    @pytest.mark.asyncio
    async def test_chunk_flush_rotates_open_non_chunk_file(self, wal_dir):
        """Non-chunk write followed by a chunk flush must not truncate the non-chunk file.

        _get_current_filename() uses a second-resolution timestamp + _file_seq.
        Without an explicit rotation before the chunk flush, both the non-chunk
        file and the chunk file would share the same name on fast hardware, and
        the 'w'-mode chunk open would silently destroy the non-chunk data.
        """
        writer = WalWriter(wal_dir)

        # Non-chunk write: opens a file at _file_seq=0
        await writer.log_mutation('CREATE (n:Before)', {}, 'db')

        # Chunk: should rotate the open file and flush to a separate new file
        async with writer.chunk():
            for i in range(3):
                await writer.log_mutation(f'CREATE (n:Chunk{i})', {}, 'db')

        await writer.close()

        files = sorted(wal_dir.glob('*.jsonl'))
        # Must be exactly 2 files: one for the non-chunk write, one for the chunk
        assert len(files) == 2, f'Expected 2 files but got {len(files)}: {files}'

        def read_cyphers(path):
            with open(path) as f:
                return [json.loads(line)['cypher'] for line in f if line.strip()]

        cyphers0 = read_cyphers(files[0])
        cyphers1 = read_cyphers(files[1])

        # First file must contain only the non-chunk write (not truncated by the chunk)
        assert cyphers0 == ['CREATE (n:Before)'], f'Non-chunk file corrupted: {cyphers0}'
        # Second file must contain all chunk mutations
        assert len(cyphers1) == 3, f'Chunk file has wrong line count: {cyphers1}'


class TestSerializeValueEmbeddingCompression:
    """Test that _serialize_value compresses long numeric lists as f16: base64 strings."""

    def test_long_float_list_produces_f16_string(self):
        vec = [float(i) / 1000.0 for i in range(768)]
        result = WalWriter._serialize_value(vec)
        assert isinstance(result, str)
        assert result.startswith('f16:')

    def test_short_float_list_not_encoded(self):
        vec = [0.1, 0.2, 0.3]
        result = WalWriter._serialize_value(vec)
        assert isinstance(result, list)

    def test_exactly_64_elements_not_encoded(self):
        """Threshold is > 64, so exactly 64 elements should NOT be encoded."""
        vec = [float(i) for i in range(64)]
        result = WalWriter._serialize_value(vec)
        assert isinstance(result, list)

    def test_65_elements_is_encoded(self):
        """65 elements exceeds the threshold and SHOULD be encoded."""
        vec = [float(i) for i in range(65)]
        result = WalWriter._serialize_value(vec)
        assert isinstance(result, str)
        assert result.startswith('f16:')

    def test_long_string_list_not_encoded(self):
        """Long lists of strings must pass through unchanged (no compression)."""
        strings = ['word'] * 100
        result = WalWriter._serialize_value(strings)
        assert isinstance(result, list)
        assert result == strings

    def test_roundtrip_within_float16_tolerance(self):
        """Serialize → deserialize round-trip stays within float16 precision."""
        vec = [float(i) / 1000.0 for i in range(768)]
        encoded = WalWriter._serialize_value(vec)
        assert isinstance(encoded, str)
        decoded = decode_embedding_param(encoded)
        assert isinstance(decoded, list)
        assert len(decoded) == 768
        for orig, dec in zip(vec, decoded):
            assert abs(orig - dec) < 0.001

    @pytest.mark.asyncio
    async def test_log_mutation_embeds_f16_in_wal_file(self, tmp_path):
        """End-to-end: embedding params are written as f16: strings in JSONL."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        writer = WalWriter(str(wal_dir))
        vec = [float(i) / 1000.0 for i in range(768)]
        await writer.log_mutation(
            'MERGE (n:Entity {uuid: $uuid}) SET n.name_embedding = $name_embedding',
            {'uuid': 'abc', 'name_embedding': vec},
            'testdb',
        )
        await writer.close()

        wal_files = list(wal_dir.glob('*.jsonl'))
        assert len(wal_files) == 1
        with open(wal_files[0]) as f:
            entry = json.loads(f.readline())
        assert entry['params']['uuid'] == 'abc'
        assert isinstance(entry['params']['name_embedding'], str)
        assert entry['params']['name_embedding'].startswith('f16:')
