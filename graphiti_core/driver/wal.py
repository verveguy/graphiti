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

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default max events per WAL file before rotation (~1MB per file at ~20KB/event).
# Keeps the active tip file small for git-friendly commits.
DEFAULT_MAX_EVENTS_PER_FILE = 50

# Mutation keywords that indicate a write operation (case-insensitive)
_MUTATION_KEYWORDS = frozenset({'CREATE', 'MERGE', 'SET', 'DELETE', 'DETACH', 'DROP', 'REMOVE'})

# Regex to strip string literals (single and double quoted, with escape handling)
_STRING_LITERAL_PATTERN = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# Index DDL patterns (case-insensitive)
_INDEX_DDL_PATTERNS = [
    re.compile(r'\bCREATE\s+(UNIQUE\s+)?INDEX\b', re.IGNORECASE),
    re.compile(r'\bCREATE\s+FULLTEXT\s+INDEX\b', re.IGNORECASE),
    re.compile(r'\bCREATE\s+VECTOR\s+INDEX\b', re.IGNORECASE),
    re.compile(r'\bDROP\s+(FULLTEXT\s+)?INDEX\b', re.IGNORECASE),
    re.compile(r'\bCALL\s+db\.idx\.fulltext\b', re.IGNORECASE),
    re.compile(r'\bCALL\s+db\.indexes\b', re.IGNORECASE),
]


class WalWriter:
    """
    Write-ahead log for capturing FalkorDB mutations as replayable JSONL files.

    Each WAL entry contains:
    - seq: Monotonically increasing sequence number (global across all files)
    - ts: ISO 8601 timestamp of when the mutation was logged
    - db: Database name the mutation was executed against
    - cypher: The Cypher query that was executed
    - params: Query parameters (with datetimes converted to strings)

    Files are rotated after max_events_per_file entries. File naming follows:
    {timestamp}_{session_uuid_short}_{file_seq:04d}.jsonl
    """

    def __init__(self, wal_dir: str | Path, max_events_per_file: int = DEFAULT_MAX_EVENTS_PER_FILE):
        """
        Initialize the WAL writer.

        Args:
            wal_dir: Directory to write WAL files to. Created if it doesn't exist.
            max_events_per_file: Maximum number of events per file before rotation.
        """
        self._wal_dir = Path(wal_dir)
        self._max_events = max_events_per_file
        self._lock = asyncio.Lock()
        self._file: Any = None  # File handle
        self._current_seq = 0
        self._events_in_file = 0
        self._session_id = uuid.uuid4().hex[:6]
        self._file_seq = 0
        self._closed = False
        self._ref_count = 1  # Reference counting for shared WAL instances

        # Create WAL directory if it doesn't exist
        self._wal_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing files for sequence continuity
        self._scan_existing_files()

    def _scan_existing_files(self) -> None:
        """Scan existing WAL files to determine the next sequence number."""
        max_seq = -1
        max_file_seq = -1

        for wal_file in sorted(self._wal_dir.glob('*.jsonl')):
            # Extract file sequence from filename
            try:
                parts = wal_file.stem.split('_')
                if len(parts) >= 3:
                    file_seq = int(parts[-1])
                    max_file_seq = max(max_file_seq, file_seq)
            except (ValueError, IndexError):
                pass

            # Read last line to get max sequence
            try:
                with open(wal_file, 'rb') as f:
                    # Seek to end and read backwards to find last line
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
                    if file_size == 0:
                        continue

                    # Read last chunk (usually enough for one line)
                    chunk_size = min(8192, file_size)
                    f.seek(file_size - chunk_size)
                    chunk = f.read()

                    # Find last complete line
                    lines = chunk.split(b'\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                if 'seq' in entry:
                                    max_seq = max(max_seq, entry['seq'])
                                    break
                            except json.JSONDecodeError:
                                continue
            except OSError as e:
                logger.warning(f'Error reading WAL file {wal_file}: {e}')

        # Set next sequence number and file sequence
        self._current_seq = max_seq + 1
        self._file_seq = max_file_seq + 1

        if max_seq >= 0:
            logger.info(
                f'WAL: Resuming from sequence {self._current_seq}, file seq {self._file_seq}'
            )

    @staticmethod
    def is_mutation(cypher: str) -> bool:
        """
        Detect if a Cypher query is a mutation (write operation).

        Checks for mutation keywords (CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE)
        after stripping string literals to avoid false positives from values.

        Args:
            cypher: The Cypher query to check.

        Returns:
            True if the query contains mutation keywords, False otherwise.
        """
        # Strip string literals to avoid matching keywords in values
        stripped = _STRING_LITERAL_PATTERN.sub('', cypher)

        # Tokenize and check for mutation keywords (case-insensitive)
        tokens = re.findall(r'\b[A-Za-z]+\b', stripped)
        return any(token.upper() in _MUTATION_KEYWORDS for token in tokens)

    @staticmethod
    def is_index_ddl(cypher: str) -> bool:
        """
        Detect if a Cypher query is an index DDL operation.

        Index operations are excluded from WAL since they're handled separately
        by build_indices_and_constraints() during replay.

        Args:
            cypher: The Cypher query to check.

        Returns:
            True if the query is an index DDL operation, False otherwise.
        """
        return any(pattern.search(cypher) for pattern in _INDEX_DDL_PATTERNS)

    def _get_current_filename(self) -> Path:
        """Generate the current WAL filename."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        return self._wal_dir / f'{timestamp}_{self._session_id}_{self._file_seq:04d}.jsonl'

    def _ensure_file_open(self) -> None:
        """Ensure we have an open file handle, creating a new file if needed.

        Uses synchronous I/O — safe for local filesystem (microsecond-scale).
        Would block the event loop on NFS or slow mounts.
        """
        if self._file is None:
            filename = self._get_current_filename()
            # File stays open for multiple writes and is closed explicitly via close()
            self._file = open(filename, 'a', encoding='utf-8')  # noqa: SIM115
            self._events_in_file = 0
            logger.debug(f'WAL: Opened new file {filename}')

    def _rotate_file(self) -> None:
        """Close current file and prepare for a new one."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._file_seq += 1
            logger.debug(f'WAL: Rotated to file seq {self._file_seq}')

    async def log_mutation(self, cypher: str, params: dict[str, Any], database: str) -> None:
        """
        Log a mutation to the WAL if it passes filtering.

        Only logs queries that:
        1. Are mutations (contain mutation keywords)
        2. Are NOT index DDL operations

        Args:
            cypher: The Cypher query that was executed.
            params: Query parameters (will be serialized to JSON).
            database: Database name the query was executed against.
        """
        if self._closed:
            logger.warning('WAL: Attempted to log to closed WAL writer')
            return

        # Filter: only log mutations, exclude index DDL
        if not self.is_mutation(cypher) or self.is_index_ddl(cypher):
            return

        async with self._lock:
            if self._closed:
                return

            # Rotate if needed
            if self._events_in_file >= self._max_events:
                self._rotate_file()

            # Ensure file is open
            self._ensure_file_open()

            # Build entry
            entry = {
                'seq': self._current_seq,
                'ts': datetime.now(timezone.utc).isoformat(),
                'db': database,
                'cypher': cypher,
                'params': self._serialize_params(params),
            }

            # Write entry
            try:
                line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
                self._file.write(line + '\n')
                self._file.flush()  # Ensure durability
                self._current_seq += 1
                self._events_in_file += 1
            except (TypeError, ValueError) as e:
                logger.error(f'WAL: Failed to serialize entry: {e}')
                raise

    @staticmethod
    def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize parameters for JSON storage.

        Handles datetime objects by converting to ISO format strings.
        Lists (including embeddings) are kept as-is since JSON handles them natively.
        """
        result = {}
        for key, value in params.items():
            result[key] = WalWriter._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Recursively serialize a value for JSON storage."""
        if isinstance(value, dict):
            return {k: WalWriter._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [WalWriter._serialize_value(item) for item in value]
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return value

    def acquire(self) -> None:
        """
        Increment the reference count for shared WAL instances.

        Call this when sharing the WAL writer with a cloned driver so that
        close() only truly closes when the last user releases its reference.
        """
        if self._closed:
            raise RuntimeError('Cannot acquire a closed WAL writer')
        self._ref_count += 1
        logger.debug(f'WAL: Acquired reference, ref_count={self._ref_count}')

    async def close(self) -> None:
        """
        Release one reference to the WAL writer.

        The underlying file is only closed when the last reference is released.
        This method is idempotent - calling it multiple times is safe (each call
        decrements the reference count at most once, and extra calls after the
        count reaches zero are no-ops).
        """
        async with self._lock:
            if self._closed:
                return

            self._ref_count -= 1
            logger.debug(f'WAL: Released reference, ref_count={self._ref_count}')

            if self._ref_count > 0:
                return

            self._closed = True

            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except OSError as e:
                    logger.error(f'WAL: Error closing file: {e}')
                finally:
                    self._file = None

            logger.debug(f'WAL: Closed writer, final seq {self._current_seq}')

    @property
    def current_sequence(self) -> int:
        """Get the current sequence number (next to be written)."""
        return self._current_seq

    @property
    def is_closed(self) -> bool:
        """Check if the WAL writer has been closed."""
        return self._closed
