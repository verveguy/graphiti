# ADR 004: WAL Embedding Encoding — Prefix-Tagged Base64 with Forever-Decode Invariant

**Status:** Accepted  
**Date:** 2026-04-30  
**Branch:** fabrik/issue-70

---

## Context

WAL JSONL files serialize embedding vectors (768-dim float arrays) as JSON float arrays. At ~12 ASCII characters per float, a single embedding field costs ~6 KB of text. In the RFC-ADR corpus, 174 ingested documents produced 1.1 GB of WAL across 17,160 files — overwhelmingly from embedding serialization. This makes WAL-based distribution (in-repo WAL trees pushed for downstream replay) impractical.

Two sub-problems had to be solved simultaneously:

1. **Write format**: How to encode embeddings compactly in new WAL entries.
2. **Read format**: How the decoder handles old WAL files (legacy arrays), new WAL files (compressed), and any future formats — without a global WAL version field.

---

## Decision

### Write format: unconditional `f16:` base64 encoding

All new WAL entries encode long float lists (> 64 elements) as `"f16:<base64>"` strings, where the payload is the raw bytes of the vector cast to `numpy.float16` and then base64-encoded. This yields ~65% size reduction vs. JSON float arrays at ~0.0005 cosine drift for normalized vectors.

- There is **no config knob**. `f16:` is the unconditional write default.
- There is **no `wal_version` field** in WAL entries.

### Read format: per-record auto-detection via prefix tags

The decoder (`decode_embedding_param` in `wal_replay_helpers.py`) inspects each value individually:

| Value type | Action |
|---|---|
| `list` | Returned as-is (legacy JSON-array format) |
| `str` starting with `"f16:"` | Decoded from float16 bytes → `list[float]` |
| `str` starting with `"f32:"` | Decoded from float32 bytes → `list[float]` (compatibility) |
| Any other string | Returned as-is |
| Any other type | Returned as-is |

This per-record, per-value approach means individual WAL files can contain a mix of formats (e.g., legacy entries from before this change alongside new compressed entries), and each line is decoded correctly without global state.

### Forever-decode invariant

**The decoder must handle all three formats (JSON array, `f16:`, `f32:`) in perpetuity.**

This is a hard constraint, not a preference. Once a format appears in a WAL file, it must remain decodable by all future versions of the replay paths. Removing or narrowing the decoder would silently corrupt replays of older WAL trees.

Concretely:
- Do not remove the `list` passthrough branch.
- Do not remove the `f16:` decode branch.
- Do not remove the `f32:` decode branch.
- New format prefixes (e.g., `f8:`, `bf16:`) may be added to the decoder, but existing branches must not be removed.

---

## Alternatives Considered

### `wal_version` field in WAL entries

A `wal_version` field in each WAL entry (or WAL file header) would let the decoder switch behavior globally. This was rejected because:

- It adds schema complexity.
- Mixed WAL trees (files from before and after the change) would require per-file version tracking.
- Per-record auto-detection is simpler and more robust: each value carries its own format indicator.

### Config knob for write format

A `wal_compression` flag to choose `none`, `f16`, or `f32` was considered. This was rejected because it adds operational complexity with no practical benefit — the precision loss from f16 is negligible for the cosine similarity use cases that Graphiti targets, and operators who need higher precision can use the `f32:` format in the migration script.

### Float32 as the write default

`f32:` encoding (30% size reduction, zero precision loss) was considered as a more conservative default. `f16:` was chosen because the ~65% reduction is significantly better and the precision loss is below the noise floor for all graphiti use cases (semantic similarity, deduplication).

---

## Consequences

### Positive

- ~65% reduction in WAL file size for typical 768-dim embedding workloads.
- WAL trees are now practical for in-repo distribution and downstream replay.
- Existing WAL files remain fully replayable with no migration required (though `wal-compress` is provided for operators who want to compress existing trees).
- The prefix-tag design is extensible: new encoding formats can be added without breaking existing files.

### Negative / constraints

- Float16 precision: ~0.0005 cosine drift on normalized vectors. Operators with strict precision requirements have no opt-out.
- The forever-decode invariant constrains future refactors: the three decoder branches must never be removed, even if the write format changes again.
- `wal_replay.py` (FalkorDB path) now runs `decode_embedding_param` on every WAL entry's params, adding a small per-entry overhead. This is negligible vs. the network I/O cost of the FalkorDB `graph.query()` call.

### Migration

The `wal-compress` CLI tool (`graphiti_core.driver.wal_compress`) rewrites existing WAL trees in-place using the `f16:` format. It is idempotent and supports `--dry-run` and `--keep-backup`. Existing WAL files work without migration — migration is optional.
