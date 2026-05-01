"""WAL embedding compression migration script.

Rewrites existing WAL JSONL trees in-place, converting legacy JSON float-array
embeddings to the f16: base64 format used by all new WAL writes.

Usage:
    wal-compress --wal-dir /path/to/wal
    wal-compress --wal-dir /path/to/wal --dry-run
    wal-compress --wal-dir /path/to/wal --keep-backup

Flags:
    --wal-dir     Directory containing *.jsonl WAL files (required)
    --dry-run     Print expected size reduction without writing
    --keep-backup Rename original to <file>.legacy before overwriting

The script is idempotent: lines already in f16: or f32: form are left
unchanged. Atomic write (temp file + os.replace) ensures a crash leaves
the original intact.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from graphiti_core.driver.wal_replay_helpers import encode_embedding

logger = logging.getLogger(__name__)


def _compress_params(params: Any) -> tuple[Any, bool]:
    """Recursively compress embedding params.

    Returns (new_value, changed) where changed is True if any encoding was done.
    """
    if isinstance(params, dict):
        new_dict = {}
        changed = False
        for k, v in params.items():
            new_v, v_changed = _compress_params(v)
            new_dict[k] = new_v
            if v_changed:
                changed = True
        return new_dict, changed
    if isinstance(params, list):
        # Already-decoded list: compress if it's a long numeric list
        if len(params) > 64 and isinstance(params[0], (int, float)):
            return encode_embedding(params), True
        return params, False
    if isinstance(params, str):
        # Already compressed — skip (idempotent)
        if params.startswith('f16:') or params.startswith('f32:'):
            return params, False
        return params, False
    return params, False


def _compress_line(line: str) -> tuple[str, int, bool]:
    """Parse a JSONL line, compress its embedding params, return (new_line, bytes_saved, changed).

    Returns the original line unchanged if it cannot be parsed or needs no changes.
    """
    stripped = line.strip()
    if not stripped:
        return line, 0, False

    try:
        entry = json.loads(stripped)
    except json.JSONDecodeError:
        return line, 0, False

    if not isinstance(entry, dict):
        return line, 0, False

    params = entry.get('params', {})
    if not params:
        return line, 0, False

    new_params, changed = _compress_params(params)
    if not changed:
        return line, 0, False

    entry['params'] = new_params
    new_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n'
    bytes_saved = len(line.encode('utf-8')) - len(new_line.encode('utf-8'))
    return new_line, bytes_saved, True


def compress_wal_dir(
    wal_dir: Path,
    dry_run: bool = False,
    keep_backup: bool = False,
) -> tuple[int, int]:
    """Compress all WAL files in wal_dir.

    Returns (total_files_modified, total_bytes_saved).
    """
    wal_files = sorted(wal_dir.glob('*.jsonl'))
    if not wal_files:
        logger.info('No WAL files found in %s', wal_dir)
        return 0, 0

    total_files_modified = 0
    total_bytes_saved = 0

    for wal_file in wal_files:
        with open(wal_file, encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False
        file_bytes_saved = 0

        for line in lines:
            new_line, bytes_saved, changed = _compress_line(line)
            new_lines.append(new_line)
            if changed:
                file_changed = True
                file_bytes_saved += bytes_saved

        if not file_changed:
            logger.debug('No changes needed: %s', wal_file.name)
            continue

        total_files_modified += 1
        total_bytes_saved += file_bytes_saved

        if dry_run:
            print(f'  {wal_file.name}: would save {file_bytes_saved:,} bytes')
            continue

        if keep_backup:
            backup_path = wal_file.with_suffix('.jsonl.legacy')
            wal_file.rename(backup_path)
            logger.debug('Backed up %s → %s', wal_file.name, backup_path.name)

        # Atomic write: temp file in the same directory so os.replace() is
        # same-filesystem and therefore atomic on POSIX.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=wal_file.parent, suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tmp_f:
                tmp_f.writelines(new_lines)
            os.replace(tmp_path, wal_file)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        print(f'  {wal_file.name}: saved {file_bytes_saved:,} bytes')

    return total_files_modified, total_bytes_saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compress WAL embeddings from JSON float arrays to f16: base64 strings',
    )
    parser.add_argument(
        '--wal-dir',
        required=True,
        type=Path,
        help='Directory containing WAL .jsonl files',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print expected size reduction without writing any files',
    )
    parser.add_argument(
        '--keep-backup',
        action='store_true',
        help='Rename original files to <file>.legacy before overwriting',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    if not args.wal_dir.is_dir():
        print(f'Error: WAL directory not found: {args.wal_dir}', file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print('DRY RUN — no files will be modified')

    print(f'Compressing WAL files in {args.wal_dir} ...')

    try:
        files_modified, bytes_saved = compress_wal_dir(
            wal_dir=args.wal_dir,
            dry_run=args.dry_run,
            keep_backup=args.keep_backup,
        )
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    action = 'would save' if args.dry_run else 'saved'
    print(
        f'\nDone: {files_modified} file(s) modified, {action} {bytes_saved:,} bytes total'
    )


if __name__ == '__main__':
    main()
