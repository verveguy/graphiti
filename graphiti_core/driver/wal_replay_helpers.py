"""Pure-Python helpers shared across WAL replay paths.

These helpers are intentionally dependency-free (no database client
imports) so they can be unit-tested without pulling in native extensions
like `real_ladybug` or `falkordb` at test collection time.
"""

from __future__ import annotations

import base64
import re
from typing import Any

import numpy as np

# WAL entries produced by wal_dump.py (the one-time FalkorDB → WAL dump
# path) wrap vector parameters in `vecf32($ident)` because that's the
# FalkorDB Cypher syntax. LadybugDB has no `vecf32()` function — its
# `FLOAT[N]` column type handles the cast implicitly when the parameter
# is supplied as a plain list[float]. This regex rewrites
# `vecf32($x)` or `vecf32($x.y)` back to the bare parameter reference
# `$x` / `$x.y` before execution, so WAL dumps from the FalkorDB era
# can be replayed against a LadybugDB target.
#
# Live writes from LadybugDriver against LadybugDB never emit vecf32()
# (the provider-switched model queries in graphiti_core/models/ have a
# LADYBUG branch that uses `$name_embedding` directly), so on pure
# LadybugDB-era WAL this substitution is a no-op.
_VECF32_WRAPPER_RE = re.compile(
    r'\bvecf32\(\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\)'
)


def strip_vecf32_wrappers(cypher: str) -> str:
    """Remove `vecf32($ident)` wrappers from a Cypher query string.

    Used by `replay_wal_ladybug()` to normalize FalkorDB-era WAL entries
    for execution against LadybugDB. Returns the original string
    unchanged when no wrappers are present (idempotent / no-op safe).
    """
    return _VECF32_WRAPPER_RE.sub(r'$\1', cypher)


# FalkorDB / Neo4j support bulk property set from a map parameter:
#   SET n = $props
# where $props is a dict like {"name": "...", "content": "...", ...}.
# LadybugDB does not support this syntax — it requires
# individual property assignments:
#   SET n.name = $name, n.content = $content, ...
#
# This regex detects `SET <var> = $<param>` patterns. It must NOT
# match `SET n.field = $value` (individual assignments), so we
# require no dot after the variable name.
_BULK_SET_RE = re.compile(r'SET\s+([a-zA-Z_]\w*)\s*=\s*\$([a-zA-Z_]\w*)(?=\s|$|,)')


def expand_bulk_property_set(
    cypher: str,
    params: dict,
) -> tuple[str, dict]:
    """Expand `SET n = $props` into individual property assignments.

    FalkorDB-era WAL entries use Neo4j's bulk property set syntax where
    a single dict parameter sets all properties at once. LadybugDB does not
    support this — it needs `SET n.field1 = $field1, n.field2 = $field2`.

    This function:
    1. Finds `SET <var> = $<param>` in the Cypher string
    2. Looks up params[<param>] — if it's a dict, expands it
    3. Replaces the SET clause with individual assignments
    4. Flattens the nested dict into top-level params

    Returns (cypher, params) — both potentially modified. If no bulk SET
    is found, or the param isn't a dict, returns the inputs unchanged.

    Handles multiple bulk SET clauses in the same query (e.g., both node
    and edge properties), and mixed bulk + individual SET assignments.
    """
    matches = list(_BULK_SET_RE.finditer(cypher))
    if not matches:
        return cypher, params

    # Work backwards through matches so replacements don't shift offsets
    new_params = dict(params)
    new_cypher = cypher

    for match in reversed(matches):
        var_name = match.group(1)  # e.g., "n" or "e"
        param_name = match.group(2)  # e.g., "props"

        if param_name not in new_params:
            continue
        props_dict = new_params[param_name]
        if not isinstance(props_dict, dict):
            continue

        # Build individual assignments: n.field1 = $field1, n.field2 = $field2
        assignments = []
        for key in props_dict:
            # Prefix with param_name to avoid collisions with existing
            # top-level params (e.g., "uuid" exists at both levels)
            flat_key = f'{param_name}_{key}'
            assignments.append(f'{var_name}.{key} = ${flat_key}')
            new_params[flat_key] = props_dict[key]

        if not assignments:
            continue

        # Replace the bulk SET with expanded assignments
        expanded = 'SET ' + ',\n    '.join(assignments)
        new_cypher = new_cypher[: match.start()] + expanded + new_cypher[match.end() :]

        # Remove the original nested dict param
        del new_params[param_name]

    return new_cypher, new_params


# ---------------------------------------------------------------------------
# WAL embedding compression helpers
#
# Embedding vectors are encoded as prefix-tagged base64 strings in WAL records
# to reduce file size (~65% smaller than JSON float arrays at float16).
#
# Write format (unconditional): "f16:<base64>" — float16 binary, base64-encoded.
# Decode supports: "f16:", "f32:", and legacy JSON arrays (list[float]).
#
# FOREVER-DECODE INVARIANT: this decoder must handle all three formats
# indefinitely — removing or narrowing it would break replay of older WAL trees.
# ---------------------------------------------------------------------------


def encode_embedding(vec: list[float]) -> str:
    """Encode a float list as a float16 base64 string for WAL storage.

    Returns a string of the form ``"f16:<base64>"`` where the base64 payload
    is the raw bytes of the vector cast to float16 (little-endian, as numpy
    default). Use ``decode_embedding_param`` to recover the original values.
    """
    arr = np.array(vec, dtype=np.float16)
    return 'f16:' + base64.b64encode(arr.tobytes()).decode('ascii')


def decode_embedding_param(value: Any) -> Any:
    """Decode a WAL parameter value, handling all embedding encoding formats.

    Auto-detects the format of each value:
    - ``list`` → returned as-is (legacy JSON-array format)
    - ``str`` starting with ``"f16:"`` → decoded from float16 bytes → ``list[float]``
    - ``str`` starting with ``"f32:"`` → decoded from float32 bytes → ``list[float]``
    - ``dict`` → recurses into values
    - all other types → returned as-is

    This function satisfies the forever-decode invariant: it handles all three
    formats and must continue to do so as long as WAL replay exists.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value.startswith('f16:'):
            raw = base64.b64decode(value[4:])
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).tolist()
        if value.startswith('f32:'):
            raw = base64.b64decode(value[4:])
            return np.frombuffer(raw, dtype=np.float32).tolist()
        return value
    if isinstance(value, dict):
        return {k: decode_embedding_param(v) for k, v in value.items()}
    return value
