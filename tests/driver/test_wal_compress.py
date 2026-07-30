"""Tests for the WAL compression migration script (wal_compress.py).

Tests cover:
- Basic compression of legacy JSON float arrays to f16: base64 strings
- Idempotency: already-compressed files are not modified on second run
- --dry-run: no files written, correct byte savings reported
- --keep-backup: original files renamed to .legacy
- Mixed files: legacy and already-compressed lines in the same file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphiti_core.driver.wal_compress import compress_wal_dir
from graphiti_core.driver.wal_replay_helpers import decode_embedding_param, encode_embedding


def _write_wal_file(wal_dir: Path, filename: str, entries: list[dict]) -> Path:
    path = wal_dir / filename
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
    return path


def _make_entry(seq: int, vec: list[float] | str, uuid: str = 'x') -> dict:
    return {
        'seq': seq,
        'ts': '2026-03-24T00:00:00Z',
        'db': 'db',
        'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n.emb = $emb',
        'params': {'uuid': uuid, 'emb': vec},
    }


def _vec(n: int = 768) -> list[float]:
    return [float(i) / 1000.0 for i in range(n)]


class TestCompressWalDirBasic:
    def test_compresses_legacy_embeddings(self, tmp_path):
        """Legacy list[float] embeddings are rewritten as f16: strings."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        vec = _vec()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, vec)])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)

        assert files_modified == 1
        assert bytes_saved > 0

        # Verify file contains f16: encoding
        lines = (wal_dir / '0001.jsonl').read_text().splitlines()
        entry = json.loads(lines[0])
        assert isinstance(entry['params']['emb'], str)
        assert entry['params']['emb'].startswith('f16:')

    def test_decoded_values_within_float16_tolerance(self, tmp_path):
        """After compression, decoded embeddings are within float16 precision of originals."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        vec = _vec()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, vec)])

        compress_wal_dir(wal_dir)

        lines = (wal_dir / '0001.jsonl').read_text().splitlines()
        entry = json.loads(lines[0])
        decoded = decode_embedding_param(entry['params']['emb'])
        assert len(decoded) == 768
        for orig, dec in zip(vec, decoded):
            assert abs(orig - dec) < 0.001

    def test_non_embedding_params_unchanged(self, tmp_path):
        """Short lists, strings, and non-numeric params pass through unchanged."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        entry = {
            'seq': 0,
            'ts': '2026-03-24T00:00:00Z',
            'db': 'db',
            'cypher': 'CREATE (n:Test {uuid: $uuid, name: $name})',
            'params': {'uuid': 'abc', 'name': 'Alice', 'tags': ['a', 'b', 'c']},
        }
        _write_wal_file(wal_dir, '0001.jsonl', [entry])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)

        assert files_modified == 0  # No embedding fields — file unchanged
        assert bytes_saved == 0

    def test_empty_directory(self, tmp_path):
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 0
        assert bytes_saved == 0


class TestIdempotency:
    def test_second_run_makes_no_changes(self, tmp_path):
        """Running compress twice on the same dir is idempotent."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, _vec())])

        files_modified_1, bytes_saved_1 = compress_wal_dir(wal_dir)
        assert files_modified_1 == 1
        assert bytes_saved_1 > 0

        size_after_first = (wal_dir / '0001.jsonl').stat().st_size

        files_modified_2, bytes_saved_2 = compress_wal_dir(wal_dir)
        assert files_modified_2 == 0
        assert bytes_saved_2 == 0

        size_after_second = (wal_dir / '0001.jsonl').stat().st_size
        assert size_after_first == size_after_second

    def test_already_f16_file_skipped(self, tmp_path):
        """Files with f16: embeddings are not modified."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        vec = _vec()
        f16_encoded = encode_embedding(vec)
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, f16_encoded)])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 0
        assert bytes_saved == 0

    def test_already_f32_file_skipped(self, tmp_path):
        """Files with f32: embeddings are not modified."""
        import base64
        import numpy as np

        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        vec = _vec()
        arr = np.array(vec, dtype=np.float32)
        f32_encoded = 'f32:' + base64.b64encode(arr.tobytes()).decode('ascii')
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, f32_encoded)])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 0
        assert bytes_saved == 0


class TestDryRun:
    def test_dry_run_does_not_modify_files(self, tmp_path):
        """--dry-run reports savings but writes nothing."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, _vec())])

        original_content = (wal_dir / '0001.jsonl').read_bytes()

        files_modified, bytes_saved = compress_wal_dir(wal_dir, dry_run=True)

        assert files_modified == 1
        assert bytes_saved > 0
        # File must be unchanged
        assert (wal_dir / '0001.jsonl').read_bytes() == original_content

    def test_dry_run_reports_correct_savings(self, tmp_path):
        """Bytes saved in dry-run matches actual savings from real run."""
        wal_dir_dry = tmp_path / 'wal_dry'
        wal_dir_dry.mkdir()
        wal_dir_real = tmp_path / 'wal_real'
        wal_dir_real.mkdir()

        vec = _vec()
        _write_wal_file(wal_dir_dry, '0001.jsonl', [_make_entry(0, vec)])
        _write_wal_file(wal_dir_real, '0001.jsonl', [_make_entry(0, vec)])

        _, dry_bytes = compress_wal_dir(wal_dir_dry, dry_run=True)
        _, real_bytes = compress_wal_dir(wal_dir_real, dry_run=False)

        assert dry_bytes == real_bytes


class TestKeepBackup:
    def test_keep_backup_renames_original(self, tmp_path):
        """--keep-backup renames the original to .legacy."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, _vec())])

        original_content = (wal_dir / '0001.jsonl').read_bytes()

        compress_wal_dir(wal_dir, keep_backup=True)

        assert (wal_dir / '0001.jsonl.legacy').exists()
        assert (wal_dir / '0001.jsonl').exists()
        assert (wal_dir / '0001.jsonl.legacy').read_bytes() == original_content

    def test_keep_backup_new_file_is_compressed(self, tmp_path):
        """After --keep-backup, the new .jsonl file contains compressed embeddings."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        vec = _vec()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, vec)])

        compress_wal_dir(wal_dir, keep_backup=True)

        lines = (wal_dir / '0001.jsonl').read_text().splitlines()
        entry = json.loads(lines[0])
        assert entry['params']['emb'].startswith('f16:')

    def test_no_backup_on_second_idempotent_run(self, tmp_path):
        """Second run with --keep-backup on already-compressed file creates no .legacy."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()
        _write_wal_file(wal_dir, '0001.jsonl', [_make_entry(0, _vec())])

        compress_wal_dir(wal_dir, keep_backup=True)
        assert (wal_dir / '0001.jsonl.legacy').exists()

        # Remove the .legacy to check second run doesn't create another
        (wal_dir / '0001.jsonl.legacy').unlink()

        compress_wal_dir(wal_dir, keep_backup=True)
        assert not (wal_dir / '0001.jsonl.legacy').exists()


class TestMixedFile:
    def test_mixed_file_only_legacy_lines_rewritten(self, tmp_path):
        """In a file with both legacy and f16: lines, only legacy lines are changed."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        vec_a = _vec()
        vec_b = [float(i) / 500.0 for i in range(768)]
        f16_b = encode_embedding(vec_b)

        _write_wal_file(
            wal_dir,
            '0001.jsonl',
            [
                _make_entry(0, vec_a, uuid='legacy'),
                _make_entry(1, f16_b, uuid='already-compressed'),
            ],
        )

        original_line1 = json.dumps(
            _make_entry(1, f16_b, uuid='already-compressed'),
            ensure_ascii=False,
            separators=(',', ':'),
        )

        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 1
        assert bytes_saved > 0

        lines = (wal_dir / '0001.jsonl').read_text().splitlines()
        assert len(lines) == 2

        entry0 = json.loads(lines[0])
        assert entry0['params']['emb'].startswith('f16:')
        assert entry0['params']['uuid'] == 'legacy'

        entry1 = json.loads(lines[1])
        assert entry1['params']['emb'] == f16_b  # unchanged
        assert entry1['params']['uuid'] == 'already-compressed'

    def test_multiple_files_processed(self, tmp_path):
        """Multiple JSONL files are all compressed."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        for i in range(3):
            _write_wal_file(wal_dir, f'000{i}.jsonl', [_make_entry(i, _vec(), uuid=f'u{i}')])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 3
        assert bytes_saved > 0

    def test_nested_dict_params_compressed(self, tmp_path):
        """Embedding values nested inside a dict param are also compressed."""
        wal_dir = tmp_path / 'wal'
        wal_dir.mkdir()

        vec = _vec()
        entry = {
            'seq': 0,
            'ts': '2026-03-24T00:00:00Z',
            'db': 'db',
            'cypher': 'MERGE (n:Entity {uuid: $uuid}) SET n = $props',
            'params': {'uuid': 'x', 'props': {'name': 'Alice', 'emb': vec}},
        }
        _write_wal_file(wal_dir, '0001.jsonl', [entry])

        files_modified, bytes_saved = compress_wal_dir(wal_dir)
        assert files_modified == 1

        lines = (wal_dir / '0001.jsonl').read_text().splitlines()
        result = json.loads(lines[0])
        assert result['params']['props']['emb'].startswith('f16:')
        assert result['params']['props']['name'] == 'Alice'
